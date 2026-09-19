"""Versioned, privacy-conscious handoff from Agent 1 to Agent 2.

The legacy matcher accepts only ``hpo_id`` and ``status``.  This adapter keeps
those keys while retaining the confidence components, subject, review state,
source span, gestational age and modifiers required by probability-aware Agent
2 implementations.  Raw report text is deliberately excluded.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .review import require_clinician_approval
from .schemas import (
    AssertionStatus,
    HPOCandidate,
    PatientProfile,
    PhenotypeMention,
    PhenotypeSubject,
    jsonable,
)


AGENT1_TO_AGENT2_SCHEMA_VERSION = "agent1-to-agent2-v1.0"
PATIENT_PROFILE_PAYLOAD_VERSION = "patient-profile-v1.0"


class Agent1ContractError(ValueError):
    """Raised when an Agent-1 profile cannot be handed to disease ranking safely."""


def patient_profile_to_payload(profile: PatientProfile) -> dict[str, Any]:
    """Serialize the exact review-bound profile, including sensitive source text."""

    if not isinstance(profile, PatientProfile):
        raise TypeError("profile must be a PatientProfile")
    return {
        "schema_version": PATIENT_PROFILE_PAYLOAD_VERSION,
        "contains_source_text": True,
        "patient_profile": jsonable(profile),
    }


def patient_profile_from_payload(payload: Mapping[str, Any]) -> PatientProfile:
    """Restore a profile without re-running a potentially nondeterministic model."""

    if payload.get("schema_version") != PATIENT_PROFILE_PAYLOAD_VERSION:
        raise Agent1ContractError(
            f"Unsupported patient profile payload: {payload.get('schema_version')!r}"
        )
    raw_profile = payload.get("patient_profile")
    if not isinstance(raw_profile, Mapping):
        raise Agent1ContractError("patient_profile payload is missing")
    raw_mentions = raw_profile.get("mentions")
    if not isinstance(raw_mentions, list):
        raise Agent1ContractError("patient_profile.mentions must be an array")
    mentions: list[PhenotypeMention] = []
    for index, raw_mention in enumerate(raw_mentions):
        if not isinstance(raw_mention, Mapping):
            raise Agent1ContractError(f"patient_profile.mentions[{index}] must be an object")
        raw_candidates = raw_mention.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise Agent1ContractError(
                f"patient_profile.mentions[{index}].candidates must be an array"
            )
        candidates = [
            HPOCandidate(
                hpo_id=str(item["hpo_id"]),
                hpo_name=str(item["hpo_name"]),
                score=float(item["score"]),
                matched_alias=str(item.get("matched_alias", "")),
                retrieval_method=str(item.get("retrieval_method", "lexical")),
            )
            for item in raw_candidates
            if isinstance(item, Mapping)
        ]
        mentions.append(
            PhenotypeMention(
                mention_text=str(raw_mention.get("mention_text", "")),
                span_start=int(raw_mention.get("span_start", -1)),
                span_end=int(raw_mention.get("span_end", -1)),
                normalized_phrase=str(raw_mention.get("normalized_phrase", "")),
                hpo_id=(
                    None if raw_mention.get("hpo_id") in (None, "") else str(raw_mention["hpo_id"])
                ),
                hpo_name=(
                    None
                    if raw_mention.get("hpo_name") in (None, "")
                    else str(raw_mention["hpo_name"])
                ),
                assertion=AssertionStatus(str(raw_mention.get("assertion", "not_assessed"))),
                certainty=float(raw_mention.get("certainty", 1.0)),
                gestational_age_weeks=raw_mention.get("gestational_age_weeks"),
                observation_method=raw_mention.get("observation_method"),
                laterality=raw_mention.get("laterality"),
                severity=raw_mention.get("severity"),
                distribution=raw_mention.get("distribution"),
                progression=raw_mention.get("progression"),
                measurements=dict(raw_mention.get("measurements") or {}),
                candidates=candidates,
                model_confidence=raw_mention.get("model_confidence"),
                needs_review=bool(raw_mention.get("needs_review", True)),
                provenance=dict(raw_mention.get("provenance") or {}),
                extraction_confidence=raw_mention.get("extraction_confidence"),
                linking_confidence=raw_mention.get("linking_confidence"),
                subject=PhenotypeSubject.coerce(raw_mention.get("subject", "fetus")),
            )
        )
    return PatientProfile(
        case_id=str(raw_profile.get("case_id", "")),
        mentions=mentions,
        source_text=str(raw_profile.get("source_text", "")),
        clinician_approved=bool(raw_profile.get("clinician_approved", False)),
        reviewer=(
            None if raw_profile.get("reviewer") in (None, "") else str(raw_profile["reviewer"])
        ),
        ontology_version=(
            None
            if raw_profile.get("ontology_version") in (None, "")
            else str(raw_profile["ontology_version"])
        ),
        report_id=(
            None if raw_profile.get("report_id") in (None, "") else str(raw_profile["report_id"])
        ),
        provenance=dict(raw_profile.get("provenance") or {}),
        clinical_context=dict(raw_profile.get("clinical_context") or {}),
    )


def _reviewer_status(profile: PatientProfile, mention: PhenotypeMention) -> str:
    if (
        profile.clinician_approved
        and not mention.needs_review
        and mention.provenance.get("reviewer_status") == "approved"
    ):
        return "approved"
    return "pending"


def _observation_payload(profile: PatientProfile, mention: PhenotypeMention) -> dict[str, Any]:
    return {
        "hpo_id": mention.hpo_id,
        "hpo_name": mention.hpo_name,
        "status": mention.assertion.value,
        "mention_text": mention.mention_text,
        "source_span": {"start": mention.span_start, "end": mention.span_end},
        "subject": mention.subject.value,
        "assertion_certainty": mention.certainty,
        "extraction_confidence": mention.extraction_confidence,
        "linking_confidence": mention.linking_confidence,
        "reviewer_status": _reviewer_status(profile, mention),
        "gestational_age_weeks": mention.gestational_age_weeks,
        "observation_method": mention.observation_method,
        "modifiers": {
            "laterality": mention.laterality,
            "severity": mention.severity,
            "distribution": mention.distribution,
            "progression": mention.progression,
            "measurements": dict(mention.measurements),
        },
        "provenance": {
            "report_id": profile.report_id,
            "extractor": mention.provenance.get("extractor"),
            "matched_alias": mention.provenance.get("matched_alias"),
            "clinician_action": mention.provenance.get("clinician_action"),
        },
    }


def profile_to_agent2_request(
    profile: PatientProfile,
    *,
    require_reviewed: bool = True,
    clinical_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the versioned Agent-2 request without exposing raw report text.

    Only fetal observations are sent to disease scoring. Maternal and family
    phenotypes are retained as contextual observations so the legacy matcher
    cannot accidentally treat them as findings in the fetus.
    """

    if not isinstance(profile, PatientProfile):
        raise TypeError("profile must be a PatientProfile")
    if require_reviewed:
        require_clinician_approval(profile)

    observations: list[dict[str, Any]] = []
    contextual: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    warnings: list[str] = []

    for mention in profile.mentions:
        if mention.hpo_id is None:
            ignored.append(
                {
                    "mention_text": mention.mention_text,
                    "reason": "ungrounded_hpo",
                    "source_span": {"start": mention.span_start, "end": mention.span_end},
                }
            )
            continue
        payload = _observation_payload(profile, mention)
        if mention.subject is PhenotypeSubject.FETUS:
            observations.append(payload)
        else:
            contextual.append(payload)

    positive_fetal = [
        item
        for item in observations
        if item["status"] in {AssertionStatus.PRESENT.value, AssertionStatus.SUSPECTED.value}
    ]
    if require_reviewed and not positive_fetal:
        raise Agent1ContractError(
            "Agent 2 requires at least one clinician-approved fetal PRESENT or SUSPECTED HPO"
        )
    if contextual:
        counts = Counter(str(item["subject"]) for item in contextual)
        warnings.append(
            "Non-fetal HPO observations were retained as context and excluded from fetal disease scoring: "
            + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )

    merged_context = dict(profile.clinical_context)
    if clinical_context:
        merged_context.update(dict(clinical_context))
    gestational_ages = [
        float(item["gestational_age_weeks"])
        for item in observations
        if item.get("gestational_age_weeks") is not None
    ]
    if gestational_ages and "gestational_age_weeks" not in merged_context:
        merged_context["gestational_age_weeks"] = max(gestational_ages)

    return {
        "schema_version": AGENT1_TO_AGENT2_SCHEMA_VERSION,
        "case_id": profile.case_id,
        "ontology_version": profile.ontology_version,
        "observations": observations,
        "contextual_observations": contextual,
        "ignored_mentions": ignored,
        "clinical_context": merged_context,
        "review": {
            "approved": bool(profile.clinician_approved),
            "reviewer": profile.reviewer,
            "review_gate_version": profile.provenance.get("review_gate_version"),
            "reviewed_profile_fingerprint": profile.provenance.get(
                "reviewed_profile_fingerprint"
            ),
        },
        "confidence_semantics": {
            "assertion_certainty": "certainty of the asserted clinical state",
            "extraction_confidence": "confidence that the text span is a phenotype mention",
            "linking_confidence": "confidence that the mention is linked to this HPO term",
            "reviewer_status": "clinician review state; not a disease probability",
        },
        "warnings": warnings,
        "research_use_only": True,
        "diagnostic_claim": False,
    }


def profile_to_agent2_observations(
    profile: PatientProfile,
    *,
    require_reviewed: bool = True,
    clinical_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compatibility helper for matchers that accept an observation list."""

    return profile_to_agent2_request(
        profile,
        require_reviewed=require_reviewed,
        clinical_context=clinical_context,
    )["observations"]


__all__ = [
    "AGENT1_TO_AGENT2_SCHEMA_VERSION",
    "PATIENT_PROFILE_PAYLOAD_VERSION",
    "Agent1ContractError",
    "profile_to_agent2_observations",
    "profile_to_agent2_request",
    "patient_profile_from_payload",
    "patient_profile_to_payload",
]
