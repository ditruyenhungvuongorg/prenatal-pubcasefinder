"""Clinician review gate and portable CSV review sheet."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hpo_catalog import HPOCatalog
from .schemas import AssertionStatus, PatientProfile, PhenotypeSubject


class ClinicianApprovalRequiredError(RuntimeError):
    """Raised when downstream ranking is attempted before explicit review."""


REVIEW_FIELDS = (
    "case_id",
    "report_id",
    "profile_fingerprint",
    "mention_index",
    "mention_text",
    "span_start",
    "span_end",
    "assertion_model",
    "hpo_id_model",
    "hpo_name_model",
    "subject_model",
    "extraction_confidence_model",
    "linking_confidence_model",
    "candidates_json",
    "clinician_action",
    "clinician_hpo_id",
    "clinician_assertion",
    "clinician_subject",
    "reviewer",
    "comments",
)


REVIEW_GATE_VERSION = "review-integrity-v3"


def _sheet_value(value: Any) -> str | int | float:
    return "" if value is None else value


def profile_fingerprint(profile: PatientProfile, *, include_review_state: bool = True) -> str:
    """Return a stable digest binding a review to one exact profile revision.

    This is an integrity check against accidental cross-case or stale-sheet
    application.  It is deliberately not presented as clinician identity
    authentication or a digital signature.
    """

    payload = {
        "case_id": str(profile.case_id),
        "report_id": str(profile.report_id or ""),
        "ontology_version": str(profile.ontology_version or ""),
        "source_text_sha256": hashlib.sha256(profile.source_text.encode("utf-8")).hexdigest(),
        "mentions": [
            {
                "mention_text": mention.mention_text,
                "span_start": mention.span_start,
                "span_end": mention.span_end,
                "normalized_phrase": mention.normalized_phrase,
                "hpo_id": mention.hpo_id,
                "hpo_name": mention.hpo_name,
                "assertion": mention.assertion.value,
                "certainty": mention.certainty,
                "gestational_age_weeks": mention.gestational_age_weeks,
                "observation_method": mention.observation_method,
                "laterality": mention.laterality,
                "severity": mention.severity,
                "distribution": mention.distribution,
                "progression": mention.progression,
                "measurements": mention.measurements,
                "extraction_confidence": mention.extraction_confidence,
                "linking_confidence": mention.linking_confidence,
                "subject": mention.subject.value,
                **({"needs_review": mention.needs_review} if include_review_state else {}),
                "candidates": [
                    {
                        "hpo_id": item.hpo_id,
                        "hpo_name": item.hpo_name,
                        "score": item.score,
                        "retrieval_method": item.retrieval_method,
                    }
                    for item in mention.candidates
                ],
            }
            for mention in profile.mentions
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def review_rows_for_profile(profile: PatientProfile) -> list[dict[str, Any]]:
    """Create immutable model-side fields for an in-memory or CSV review sheet."""

    fingerprint = profile_fingerprint(profile)
    rows: list[dict[str, Any]] = []
    for index, mention in enumerate(profile.mentions):
        rows.append(
            {
                "case_id": profile.case_id,
                "report_id": profile.report_id or "",
                "profile_fingerprint": fingerprint,
                "mention_index": index,
                "mention_text": mention.mention_text,
                "span_start": mention.span_start,
                "span_end": mention.span_end,
                "assertion_model": mention.assertion.value,
                "hpo_id_model": mention.hpo_id or "",
                "hpo_name_model": mention.hpo_name or "",
                "subject_model": mention.subject.value,
                "extraction_confidence_model": _sheet_value(mention.extraction_confidence),
                "linking_confidence_model": _sheet_value(mention.linking_confidence),
                "candidates_json": json.dumps(
                    [
                        {
                            "hpo_id": item.hpo_id,
                            "hpo_name": item.hpo_name,
                            "score": item.score,
                        }
                        for item in mention.candidates
                    ],
                    ensure_ascii=False,
                ),
                "clinician_action": "",
                "clinician_hpo_id": "",
                "clinician_assertion": "",
                "clinician_subject": "",
                "reviewer": "",
                "comments": "",
            }
        )
    return rows


def export_review_csv(profile: PatientProfile, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review_rows_for_profile(profile))
    return destination


def load_review_csv(path: str | Path) -> list[dict[str, str]]:
    """Load a completed UTF-8/Excel-friendly review sheet."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(REVIEW_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Review CSV requires columns: {list(REVIEW_FIELDS)}")
        return [dict(row) for row in reader]


def apply_review_rows(
    profile: PatientProfile,
    rows: Sequence[Mapping[str, Any]],
    catalog: HPOCatalog,
) -> PatientProfile:
    """Apply accept/edit/reject decisions and return a reviewed copy."""

    reviewed = deepcopy(profile)
    source_fingerprint = profile_fingerprint(profile)
    by_index: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        index = int(row["mention_index"])
        if index in by_index:
            raise ValueError(f"Duplicate review decision for mention {index}")
        by_index[index] = row
    unexpected = sorted(set(by_index) - set(range(len(reviewed.mentions))))
    if unexpected:
        raise ValueError(f"Review decisions reference unknown mentions: {unexpected}")
    kept = []
    reviewers: set[str] = set()
    for index, mention in enumerate(reviewed.mentions):
        if index not in by_index:
            raise ClinicianApprovalRequiredError(f"Missing review decision for mention {index}")
        row = by_index[index]
        expected_model_fields = {
            "case_id": str(profile.case_id),
            "report_id": str(profile.report_id or ""),
            "profile_fingerprint": source_fingerprint,
            "mention_text": mention.mention_text,
            "assertion_model": mention.assertion.value,
            "hpo_id_model": mention.hpo_id or "",
            "hpo_name_model": mention.hpo_name or "",
            "subject_model": mention.subject.value,
            "extraction_confidence_model": str(_sheet_value(mention.extraction_confidence)),
            "linking_confidence_model": str(_sheet_value(mention.linking_confidence)),
        }
        for field, expected in expected_model_fields.items():
            actual = str(row.get(field, ""))
            if actual != expected:
                raise ValueError(
                    f"Review row {index} does not match this profile: {field} "
                    f"is {actual!r}, expected {expected!r}"
                )
        try:
            reviewed_candidates = json.loads(str(row.get("candidates_json", "")))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Review row {index} has invalid candidates_json") from exc
        expected_candidates = [
            {
                "hpo_id": item.hpo_id,
                "hpo_name": item.hpo_name,
                "score": item.score,
            }
            for item in mention.candidates
        ]
        if reviewed_candidates != expected_candidates:
            raise ValueError(
                f"Review row {index} does not match this profile: candidates_json changed"
            )
        try:
            review_start = int(row.get("span_start", -1))
            review_end = int(row.get("span_end", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Review row {index} has invalid mention span") from exc
        if (review_start, review_end) != (mention.span_start, mention.span_end):
            raise ValueError(f"Review row {index} does not match this profile: mention span changed")
        action = str(row.get("clinician_action", "")).strip().lower()
        reviewer = str(row.get("reviewer", "")).strip()
        if action not in {"accept", "edit", "reject"} or not reviewer:
            raise ClinicianApprovalRequiredError(
                f"Mention {index} requires action accept/edit/reject and reviewer"
            )
        reviewers.add(reviewer)
        if action == "reject":
            continue
        if action == "edit":
            hpo_id = str(row.get("clinician_hpo_id", "")).strip()
            assertion = str(row.get("clinician_assertion", "")).strip().lower()
            if not catalog.validate(hpo_id):
                raise ValueError(f"Reviewed HPO ID is not in pinned catalog: {hpo_id}")
            if assertion not in {status.value for status in AssertionStatus}:
                raise ValueError(f"Invalid reviewed assertion: {assertion!r}")
            mention.hpo_id = hpo_id
            mention.hpo_name = catalog.terms[hpo_id].name
            mention.assertion = AssertionStatus(assertion)
        elif mention.hpo_id is None or not catalog.validate(mention.hpo_id):
            raise ValueError(f"Cannot accept ungrounded mention {index}")
        clinician_subject = str(row.get("clinician_subject", "")).strip().lower()
        if clinician_subject:
            try:
                mention.subject = PhenotypeSubject.coerce(clinician_subject)
            except ValueError as exc:
                raise ValueError(f"Invalid reviewed phenotype subject: {clinician_subject!r}") from exc
        mention.needs_review = False
        mention.provenance = {
            **mention.provenance,
            "clinician_action": action,
            "reviewer": reviewer,
            "review_comments": str(row.get("comments", "")).strip(),
            "reviewer_status": "approved",
        }
        kept.append(mention)
    reviewed.mentions = kept
    reviewed.clinician_approved = True
    reviewed.reviewer = "; ".join(sorted(reviewers))
    decision_payload = [
        {
            "mention_index": index,
            "clinician_action": str(by_index[index].get("clinician_action", "")).strip().lower(),
            "clinician_hpo_id": str(by_index[index].get("clinician_hpo_id", "")).strip(),
            "clinician_assertion": str(by_index[index].get("clinician_assertion", "")).strip().lower(),
            "clinician_subject": str(by_index[index].get("clinician_subject", "")).strip().lower(),
            "reviewer": str(by_index[index].get("reviewer", "")).strip(),
            "comments": str(by_index[index].get("comments", "")).strip(),
        }
        for index in sorted(by_index)
    ]
    decision_digest = hashlib.sha256(
        json.dumps(decision_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    reviewed.provenance = {
        **reviewed.provenance,
        "review_gate": "completed",
        "review_gate_version": REVIEW_GATE_VERSION,
        "source_profile_fingerprint": source_fingerprint,
        "review_decisions_sha256": decision_digest,
    }
    reviewed.provenance["reviewed_profile_fingerprint"] = profile_fingerprint(
        reviewed,
        include_review_state=False,
    )
    return reviewed


def require_clinician_approval(profile: PatientProfile) -> None:
    if not profile.clinician_approved or not profile.reviewer:
        raise ClinicianApprovalRequiredError(
            "Agent 2/3 refused: a clinician must approve the Agent-1 profile first."
        )
    pending = [index for index, mention in enumerate(profile.mentions) if mention.needs_review]
    if pending:
        raise ClinicianApprovalRequiredError(f"Mentions still pending review: {pending}")
    if profile.provenance.get("review_gate_version") != REVIEW_GATE_VERSION:
        raise ClinicianApprovalRequiredError(
            "Agent 2/3 refused: profile lacks a current review integrity attestation."
        )
    recorded_fingerprint = profile.provenance.get("reviewed_profile_fingerprint")
    if not recorded_fingerprint or recorded_fingerprint != profile_fingerprint(
        profile,
        include_review_state=False,
    ):
        raise ClinicianApprovalRequiredError(
            "Agent 2/3 refused: reviewed profile changed after clinician review."
        )
    positive = [
        mention
        for mention in profile.mentions
        if mention.hpo_id
        and mention.assertion in {AssertionStatus.PRESENT, AssertionStatus.SUSPECTED}
    ]
    if not positive:
        raise ClinicianApprovalRequiredError(
            "Agent 2/3 refused: at least one clinician-approved PRESENT or SUSPECTED HPO is required."
        )
