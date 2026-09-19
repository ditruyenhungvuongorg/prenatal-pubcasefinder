"""Adapter that grounds a fine-tuned Agent-1 LLM in the pinned HPO catalog."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Mapping

from .hpo_catalog import HPOCatalog
from .schemas import AssertionStatus, PatientProfile, PhenotypeMention, PhenotypeSubject
from .training_data import SYSTEM_PROMPT_AGENT1


class InvalidModelOutputError(ValueError):
    """Raised when the LLM output is not the documented extraction schema."""


def parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```json"):
        text = text[7:].rsplit("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text[3:].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right <= left:
            raise InvalidModelOutputError("Model output contains no JSON object")
        try:
            parsed = json.loads(text[left : right + 1])
        except json.JSONDecodeError as exc:
            raise InvalidModelOutputError(f"Invalid model JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("mentions"), list):
        raise InvalidModelOutputError("Model JSON requires a mentions array")
    return parsed


class GroundedLLMAgent1:
    """Use an LLM for spans/assertions, then retrieve IDs from local HPO only.

    ``generator`` receives ``(system_prompt, report_text)`` and must return a
    string.  This keeps the adapter independent of Transformers/Unsloth and
    easy to unit-test.  Any HPO ID emitted by the model is ignored.
    """

    def __init__(
        self,
        catalog: HPOCatalog,
        generator: Callable[[str, str], str],
        *,
        top_k: int = 5,
        abstain_below: float = 0.55,
        allow_fuzzy: bool = False,
        explicit_alias_only: bool = True,
        model_version: str | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= abstain_below <= 1.0:
            raise ValueError("abstain_below must be in [0, 1]")
        self.catalog = catalog
        self.generator = generator
        self.top_k = int(top_k)
        self.abstain_below = float(abstain_below)
        self.allow_fuzzy = bool(allow_fuzzy)
        self.explicit_alias_only = bool(explicit_alias_only)
        self.model_version = model_version

    @staticmethod
    def _safe_certainty(raw_value: Any, assertion: AssertionStatus) -> tuple[float, str]:
        default = 0.5 if assertion is AssertionStatus.SUSPECTED else 1.0
        if raw_value is None:
            return default, "defaulted_missing"
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            return default, "defaulted_invalid"
        if not math.isfinite(value):
            return default, "defaulted_nonfinite"
        normalized = max(0.0, min(1.0, value))
        return normalized, "clamped" if normalized != value else "model"

    @staticmethod
    def _safe_optional_confidence(raw_value: Any) -> tuple[float | None, str]:
        if raw_value is None:
            return None, "missing"
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            return None, "invalid"
        if not math.isfinite(value):
            return None, "nonfinite"
        normalized = max(0.0, min(1.0, value))
        return normalized, "clamped" if normalized != value else "model"

    @staticmethod
    def _safe_subject(raw_value: Any) -> tuple[PhenotypeSubject, str]:
        if raw_value in (None, ""):
            return PhenotypeSubject.FETUS, "defaulted_prenatal_fetus"
        try:
            return PhenotypeSubject.coerce(str(raw_value)), "model"
        except ValueError:
            return PhenotypeSubject.UNKNOWN, "invalid_defaulted_unknown"

    def extract(
        self,
        text: str,
        *,
        case_id: str = "case-unknown",
        report_id: str | None = None,
        observation_method: str | None = None,
        clinical_context: Mapping[str, Any] | None = None,
    ) -> PatientProfile:
        raw_output = self.generator(SYSTEM_PROMPT_AGENT1, text)
        payload = parse_json_object(raw_output)
        mentions: list[PhenotypeMention] = []
        cursor = 0
        allowed_assertions = {status.value: status for status in AssertionStatus}
        for index, raw in enumerate(payload["mentions"]):
            if not isinstance(raw, Mapping):
                continue
            mention_text = str(raw.get("mention_text", "")).strip()
            if not mention_text:
                continue
            proposed_start = raw.get("span_start")
            proposed_end = raw.get("span_end")
            valid_span = (
                isinstance(proposed_start, int)
                and isinstance(proposed_end, int)
                and 0 <= proposed_start <= proposed_end <= len(text)
                and text[proposed_start:proposed_end] == mention_text
            )
            if valid_span:
                start, end = int(proposed_start), int(proposed_end)
            else:
                start = text.find(mention_text, cursor)
                if start < 0:
                    start = text.find(mention_text)
                if start < 0:
                    continue
                end = start + len(mention_text)
            cursor = max(cursor, end)
            normalized_phrase = str(raw.get("normalized_phrase") or mention_text)
            # Retrieve at least two rows even when the public top_k is one, so
            # equal-score aliases cannot be silently resolved by sort order.
            selection_candidates = [
                item
                for item in self.catalog.retrieve(
                    normalized_phrase,
                    top_k=max(self.top_k, 2),
                    allow_fuzzy=self.allow_fuzzy,
                    explicit_only=self.explicit_alias_only,
                )
                if self.catalog.validate(item.hpo_id)
                and self.catalog.terms[item.hpo_id].name == item.hpo_name
            ]
            ambiguous = bool(
                len(selection_candidates) > 1
                and math.isclose(
                    selection_candidates[0].score,
                    selection_candidates[1].score,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            candidates = selection_candidates[: self.top_k]
            selected = (
                selection_candidates[0]
                if selection_candidates
                and selection_candidates[0].score >= self.abstain_below
                and not ambiguous
                else None
            )
            assertion_text = str(raw.get("assertion", "")).strip().lower()
            assertion = allowed_assertions.get(assertion_text, AssertionStatus.NOT_ASSESSED)
            attributes = raw.get("attributes") if isinstance(raw.get("attributes"), Mapping) else {}
            measurements = attributes.get("measurements", {})
            if not isinstance(measurements, Mapping):
                measurements = {}
            certainty, certainty_status = self._safe_certainty(
                attributes.get("certainty"),
                assertion,
            )
            extraction_confidence, extraction_confidence_status = self._safe_optional_confidence(
                attributes.get("extraction_confidence", raw.get("confidence"))
            )
            subject, subject_status = self._safe_subject(attributes.get("subject"))
            if ambiguous:
                link_status = "abstain_ambiguous_top_score"
            elif selected is None:
                link_status = "abstain"
            else:
                link_status = "grounded"
            mentions.append(
                PhenotypeMention(
                    mention_text=mention_text,
                    span_start=start,
                    span_end=end,
                    normalized_phrase=normalized_phrase,
                    hpo_id=selected.hpo_id if selected else None,
                    hpo_name=selected.hpo_name if selected else None,
                    assertion=assertion,
                    certainty=certainty,
                    gestational_age_weeks=attributes.get("gestational_age_weeks"),
                    observation_method=attributes.get("observation_method") or observation_method,
                    laterality=attributes.get("laterality"),
                    severity=attributes.get("severity"),
                    distribution=attributes.get("distribution"),
                    progression=attributes.get("progression"),
                    measurements=dict(measurements),
                    candidates=candidates,
                    model_confidence=selection_candidates[0].score if selection_candidates else 0.0,
                    needs_review=True,
                    extraction_confidence=extraction_confidence,
                    linking_confidence=(
                        selection_candidates[0].score if selection_candidates else 0.0
                    ),
                    subject=subject,
                    provenance={
                        "extractor": "fine_tuned_llm_plus_grounded_retrieval_v1",
                        "model_version": self.model_version,
                        "model_mention_index": index,
                        "model_hpo_id_ignored": bool(raw.get("hpo_id")),
                        "link_status": link_status,
                        "certainty_status": certainty_status,
                        "extraction_confidence_status": extraction_confidence_status,
                        "subject_source": subject_status,
                    },
                )
            )
        grounded = sum(item.hpo_id is not None for item in mentions)
        return PatientProfile(
            case_id=case_id,
            mentions=mentions,
            source_text=text,
            clinician_approved=False,
            ontology_version=self.catalog.version,
            report_id=report_id,
            clinical_context=dict(clinical_context or {}),
            provenance={
                "agent": "agent1_llm_grounded",
                "status": "needs_review" if grounded else "abstain",
                "grounded_mentions": grounded,
                "total_mentions": len(mentions),
                "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
                "diagnostic_claim": False,
            },
        )

    run = extract


__all__ = ["GroundedLLMAgent1", "InvalidModelOutputError", "parse_json_object"]
