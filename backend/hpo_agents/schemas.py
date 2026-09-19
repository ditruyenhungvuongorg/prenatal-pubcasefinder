"""Typed contracts shared by all three agents.

The contracts deliberately keep observed, suspected, absent, and unassessable
phenotypes separate. Treating an unassessable structure as a negative finding is
a clinically meaningful error.
"""

from __future__ import annotations

import re
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


HPO_ID_RE = re.compile(r"^HP:\d{7}$")


class AssertionStatus(str, Enum):
    PRESENT = "present"
    SUSPECTED = "suspected"
    ABSENT = "absent"
    NOT_ASSESSED = "not_assessed"


class PhenotypeSubject(str, Enum):
    """Person or clinical context to which a phenotype mention belongs."""

    FETUS = "fetus"
    MOTHER = "mother"
    FAMILY_HISTORY = "family_history"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: "PhenotypeSubject | str") -> "PhenotypeSubject":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
        aliases = {
            "thai": cls.FETUS,
            "thai_nhi": cls.FETUS,
            "fetal": cls.FETUS,
            "maternal": cls.MOTHER,
            "me": cls.MOTHER,
            "family": cls.FAMILY_HISTORY,
            "familyhistory": cls.FAMILY_HISTORY,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"Invalid phenotype subject {value!r}; expected {allowed}") from exc


def _bounded_confidence(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a finite number in [0, 1]") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number in [0, 1]")
    return float(max(0.0, min(1.0, number)))


@dataclass(slots=True)
class HPOCandidate:
    hpo_id: str
    hpo_name: str
    score: float
    matched_alias: str = ""
    retrieval_method: str = "lexical"

    def __post_init__(self) -> None:
        validate_hpo_id(self.hpo_id)
        self.score = float(max(0.0, min(1.0, self.score)))


@dataclass(slots=True)
class PhenotypeMention:
    mention_text: str
    span_start: int
    span_end: int
    normalized_phrase: str
    hpo_id: str | None
    hpo_name: str | None
    assertion: AssertionStatus
    certainty: float = 1.0
    gestational_age_weeks: float | None = None
    observation_method: str | None = None
    laterality: str | None = None
    severity: str | None = None
    distribution: str | None = None
    progression: str | None = None
    measurements: dict[str, float | str] = field(default_factory=dict)
    candidates: list[HPOCandidate] = field(default_factory=list)
    model_confidence: float | None = None
    needs_review: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)
    extraction_confidence: float | None = None
    linking_confidence: float | None = None
    subject: PhenotypeSubject = PhenotypeSubject.FETUS

    def __post_init__(self) -> None:
        if self.hpo_id is not None:
            validate_hpo_id(self.hpo_id)
        if self.span_start < 0 or self.span_end < self.span_start:
            raise ValueError("Invalid mention span")
        certainty = _bounded_confidence(self.certainty, field_name="certainty")
        self.certainty = 1.0 if certainty is None else certainty
        self.model_confidence = _bounded_confidence(
            self.model_confidence,
            field_name="model_confidence",
        )
        self.extraction_confidence = _bounded_confidence(
            self.extraction_confidence,
            field_name="extraction_confidence",
        )
        self.linking_confidence = _bounded_confidence(
            self.linking_confidence,
            field_name="linking_confidence",
        )
        # ``model_confidence`` historically contained the catalog-link score.
        # Preserve that API while exposing the score with unambiguous semantics.
        if self.linking_confidence is None:
            self.linking_confidence = self.model_confidence
        if self.model_confidence is None:
            self.model_confidence = self.linking_confidence
        self.subject = PhenotypeSubject.coerce(self.subject)


@dataclass(slots=True)
class PatientProfile:
    case_id: str
    mentions: list[PhenotypeMention]
    source_text: str = ""
    clinician_approved: bool = False
    reviewer: str | None = None
    ontology_version: str | None = None
    report_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    clinical_context: dict[str, Any] = field(default_factory=dict)

    def by_status(self, status: AssertionStatus) -> list[PhenotypeMention]:
        return [m for m in self.mentions if m.assertion == status and m.hpo_id]

    @property
    def present(self) -> list[PhenotypeMention]:
        return self.by_status(AssertionStatus.PRESENT)

    @property
    def suspected(self) -> list[PhenotypeMention]:
        return self.by_status(AssertionStatus.SUSPECTED)

    @property
    def absent(self) -> list[PhenotypeMention]:
        return self.by_status(AssertionStatus.ABSENT)

    @property
    def not_assessed(self) -> list[PhenotypeMention]:
        return self.by_status(AssertionStatus.NOT_ASSESSED)

    def unique_hpo_ids(self, statuses: Iterable[AssertionStatus]) -> list[str]:
        allowed = set(statuses)
        return list(dict.fromkeys(m.hpo_id for m in self.mentions if m.hpo_id and m.assertion in allowed))


@dataclass(slots=True)
class DiseaseEvidence:
    disease_id: str
    disease_name: str
    score: float
    log_likelihood_ratio: float = 0.0
    compatibility_score: float | None = None
    confidence_score: float | None = None
    matched_hpo: list[dict[str, Any]] = field(default_factory=list)
    conflicting_hpo: list[dict[str, Any]] = field(default_factory=list)
    suspected_hpo: list[dict[str, Any]] = field(default_factory=list)
    unassessed_hpo: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)
    knowledge_base_version: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RankedDisease:
    disease_id: str
    disease_name: str
    rank: int
    score: float
    confidence_score: float | None = None
    explanation: list[str] = field(default_factory=list)
    matched_hpo: list[str] = field(default_factory=list)
    conflicting_hpo: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RankedGene:
    gene_symbol: str
    rank: int
    score: float
    matched_diseases: list[str] = field(default_factory=list)
    phenotype_evidence: list[str] = field(default_factory=list)
    inheritance_models: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TestRecommendation:
    test_name: str
    priority: int
    rationale: str
    evidence_source: str
    requires_clinician_confirmation: bool = True


def validate_hpo_id(hpo_id: str) -> str:
    if not HPO_ID_RE.fullmatch(hpo_id):
        raise ValueError(f"Invalid HPO identifier: {hpo_id!r}")
    return hpo_id


def jsonable(value: Any) -> Any:
    """Convert nested dataclasses/enums to JSON-safe Python primitives."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value
