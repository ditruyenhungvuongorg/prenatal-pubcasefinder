"""Typed input/output contract for Agent 2 (phenotype -> disease prioritisation).

This module holds the vocabulary that Agent 1, Agent 2 and downstream reviewers
share.  It deliberately separates three ideas that are easy to conflate and
clinically dangerous to conflate:

* a **ranking score** orders curated disease profiles and means nothing on an
  absolute scale;
* an **uncalibrated Bayes estimate** combines a stated prior with likelihood
  ratios but has never been checked against confirmed cases;
* a **calibrated probability** additionally required a fitted calibrator that
  was validated on held-out, clinician-confirmed cases.

Nothing in this package produces a diagnosis.  ``PhenotypeStatus`` also keeps
``ABSENT`` and ``NOT_ASSESSED`` apart, because treating an unassessable
structure as a confirmed negative is a clinically meaningful error.

Only the standard library is used so the contract can be imported anywhere,
including from notebooks that have no scientific stack installed.
"""

from __future__ import annotations

import csv
import io
import math
import re
import hashlib
import hmac
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence


HPO_ID_PATTERN = re.compile(r"^HP:\d{7}$", re.IGNORECASE)

AGENT1_TO_AGENT2_SCHEMA_VERSION = "agent1-to-agent2-v1.0"
AGENT1_TO_AGENT2_PAYLOAD_VERSION = "1.0"

#: Root of the ``Phenotypic abnormality`` sub-ontology.  It is a legal HPO term
#: and appears in official ``phenotype.hpoa`` releases, but it carries no
#: disease-discriminating information and is therefore never scored.
PHENOTYPIC_ABNORMALITY_ROOT = "HP:0000118"


class PhenotypeStatus(str, Enum):
    """Assessment state of a patient phenotype."""

    PRESENT = "PRESENT"
    SUSPECTED = "SUSPECTED"
    ABSENT = "ABSENT"
    NOT_ASSESSED = "NOT_ASSESSED"

    @classmethod
    def coerce(cls, value: "PhenotypeStatus | str") -> "PhenotypeStatus":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "POSITIVE": cls.PRESENT,
            "YES": cls.PRESENT,
            "POSSIBLE": cls.SUSPECTED,
            "UNCERTAIN": cls.SUSPECTED,
            "NEGATIVE": cls.ABSENT,
            "NO": cls.ABSENT,
            "UNKNOWN": cls.NOT_ASSESSED,
            "UNASSESSED": cls.NOT_ASSESSED,
            "NOT_EVALUATED": cls.NOT_ASSESSED,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"Unsupported phenotype status {value!r}; expected {allowed}") from exc


class ReviewerStatus(str, Enum):
    """Whether a clinician has looked at one extracted phenotype."""

    NOT_REVIEWED = "NOT_REVIEWED"
    CLINICIAN_CONFIRMED = "CLINICIAN_CONFIRMED"
    CLINICIAN_MODIFIED = "CLINICIAN_MODIFIED"
    CLINICIAN_REJECTED = "CLINICIAN_REJECTED"

    @classmethod
    def coerce(cls, value: "ReviewerStatus | str | None") -> "ReviewerStatus":
        if value is None:
            return cls.NOT_REVIEWED
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "UNREVIEWED": cls.NOT_REVIEWED,
            "PENDING": cls.NOT_REVIEWED,
            "CONFIRMED": cls.CLINICIAN_CONFIRMED,
            "APPROVED": cls.CLINICIAN_CONFIRMED,
            "ACCEPTED": cls.CLINICIAN_CONFIRMED,
            "MODIFIED": cls.CLINICIAN_MODIFIED,
            "EDITED": cls.CLINICIAN_MODIFIED,
            "CORRECTED": cls.CLINICIAN_MODIFIED,
            "REJECTED": cls.CLINICIAN_REJECTED,
            "DISCARDED": cls.CLINICIAN_REJECTED,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"Unsupported reviewer status {value!r}; expected {allowed}") from exc


class PhenotypeSubject(str, Enum):
    """Who the phenotype belongs to.

    Mirrors ``hpo_agents.schemas.PhenotypeSubject`` (Agent 1's contract) so the
    two layers stay interchangeable, but keeps ``UNKNOWN`` as the Agent 2
    default: an unstated subject must never be silently assumed to be the fetus.
    """

    FETUS = "fetus"
    MOTHER = "mother"
    FAMILY_HISTORY = "family_history"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: "PhenotypeSubject | str | None") -> "PhenotypeSubject":
        if value is None:
            return cls.UNKNOWN
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
            raise ValueError(f"Unsupported phenotype subject {value!r}; expected {allowed}") from exc


class FetalSex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: "FetalSex | str | None") -> "FetalSex":
        if value is None:
            return cls.UNKNOWN
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold()
        aliases = {
            "m": cls.MALE,
            "nam": cls.MALE,
            "f": cls.FEMALE,
            "nu": cls.FEMALE,
            "": cls.UNKNOWN,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"Unsupported fetal sex {value!r}; expected {allowed}") from exc


class ProbabilityKind(str, Enum):
    """What an ``estimated_probability`` field is actually allowed to mean."""

    #: Ordering only.  There is no prior, so no absolute number is reported.
    RANKING_SCORE = "RANKING_SCORE"
    #: prior x likelihood ratios, never checked against confirmed cases.
    BAYES_ESTIMATE_UNCALIBRATED = "BAYES_ESTIMATE_UNCALIBRATED"
    #: Bayes estimate passed through a calibrator fitted and validated on
    #: clinician-confirmed, patient-disjoint held-out cases.
    CALIBRATED_PROBABILITY = "CALIBRATED_PROBABILITY"


class ProbabilityScope(str, Enum):
    """The population/question a probability refers to."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    #: P(this disease | evidence), independently per disease.  Co-morbidity is
    #: possible, so these do not sum to 1 across the differential.
    BINARY_PER_DISEASE = "BINARY_PER_DISEASE"
    #: Relative probability inside one candidate set that assumes a single
    #: principal cause.  Includes OTHER_OR_UNKNOWN and sums to ~1.
    SINGLE_CAUSE_CANDIDATE_SET = "SINGLE_CAUSE_CANDIDATE_SET"


class BackgroundModel(str, Enum):
    """Where P(HPO | not disease) came from.

    ``ANNOTATION_BACKGROUND`` is how often a term is annotated across curated
    disease profiles.  That is *not* the frequency of the finding in people who
    do not have the disease, and any probability derived from it is an
    approximation that must be reported as such.
    """

    ANNOTATION_BACKGROUND = "annotation_background"
    COHORT_BACKGROUND = "cohort_background"
    POPULATION_BACKGROUND = "population_background"


class MissingFrequencyPolicy(str, Enum):
    """What to do when ``phenotype.hpoa`` states no frequency for an annotation."""

    #: Treat P(HPO | disease) as equal to the background: log LR = 0.  The term
    #: still counts as a semantic match but adds no likelihood evidence.
    NEUTRAL = "neutral"
    #: Use an explicitly configured default probability (documented per run).
    CONFIGURABLE_DEFAULT = "configurable_default"
    #: Drop the term from the likelihood entirely but keep the semantic match
    #: in coverage and in the explanation.
    EXCLUDE_FROM_LIKELIHOOD_BUT_KEEP_SEMANTIC_MATCH = (
        "exclude_from_likelihood_but_keep_semantic_match"
    )


class RankingStrategy(str, Enum):
    """How ``HPOAgent2Matcher.rank`` orders disease profiles.

    Both strategies read the same per-profile measures; only the sort key differs.
    """

    #: Information-content-weighted coverage, then exact coverage, then the
    #: likelihood score.  Selected on the 1,046-case PhenopacketStore validation
    #: split under leave-publication-out (top-100 recall 0.458 -> 0.620).  No
    #: prenatal validation set was available, so prenatal benefit is unproven.
    #: Absent findings and frequencies only reach the order through ``score``,
    #: i.e. among profiles with equal coverage.
    IC_COVERAGE = "ic_coverage"
    #: The original order: likelihood-dominated ``score``, then disease ID.
    LIKELIHOOD_SCORE = "likelihood_score"


class AbstentionReason(str, Enum):
    """Machine-readable reasons for refusing to produce a ranking/probability."""

    NO_VALID_HPO = "NO_VALID_HPO"
    UNKNOWN_HPO_ID = "UNKNOWN_HPO_ID"
    ONLY_ROOT_PHENOTYPE = "ONLY_ROOT_PHENOTYPE"
    ONLY_VERY_GENERAL_TERMS = "ONLY_VERY_GENERAL_TERMS"
    ALL_NOT_ASSESSED = "ALL_NOT_ASSESSED"
    UNRESOLVED_CONTRADICTION = "UNRESOLVED_CONTRADICTION"
    PROBABILITY_REQUESTED_WITHOUT_PRIOR = "PROBABILITY_REQUESTED_WITHOUT_PRIOR"
    CALIBRATOR_MISSING_OR_INCOMPATIBLE = "CALIBRATOR_MISSING_OR_INCOMPATIBLE"
    UNKNOWN_CLASS_DOMINATES = "UNKNOWN_CLASS_DOMINATES"
    LOW_CANDIDATE_COVERAGE = "LOW_CANDIDATE_COVERAGE"
    HPO_RELEASE_MISMATCH = "HPO_RELEASE_MISMATCH"
    INPUT_CONTRACT_REJECTED = "INPUT_CONTRACT_REJECTED"


class WarningCode(str, Enum):
    """Machine-readable caveats attached to a result."""

    ANNOTATION_BACKGROUND_APPROXIMATION = "ANNOTATION_BACKGROUND_APPROXIMATION"
    NAIVE_CONDITIONAL_INDEPENDENCE = "NAIVE_CONDITIONAL_INDEPENDENCE"
    RANKING_ONLY_NO_PRIOR = "RANKING_ONLY_NO_PRIOR"
    UNCALIBRATED_PROBABILITY = "UNCALIBRATED_PROBABILITY"
    SYNTHETIC_CALIBRATOR_NOT_CLINICIAN_REVIEWED = "SYNTHETIC_CALIBRATOR_NOT_CLINICIAN_REVIEWED"
    MISSING_FREQUENCY_POLICY_APPLIED = "MISSING_FREQUENCY_POLICY_APPLIED"
    ROOT_ANNOTATION_IGNORED = "ROOT_ANNOTATION_IGNORED"
    ROOT_OBSERVATION_IGNORED = "ROOT_OBSERVATION_IGNORED"
    REDUNDANT_ANCESTOR_REMOVED = "REDUNDANT_ANCESTOR_REMOVED"
    DUPLICATE_MENTION_MERGED = "DUPLICATE_MENTION_MERGED"
    PER_TERM_CONTRIBUTION_CAPPED = "PER_TERM_CONTRIBUTION_CAPPED"
    CONTEXT_RETAINED_NOT_USED = "CONTEXT_RETAINED_NOT_USED"
    DISEASE_IDENTITY_NOT_HARMONISED = "DISEASE_IDENTITY_NOT_HARMONISED"
    LOW_CANDIDATE_COVERAGE = "LOW_CANDIDATE_COVERAGE"
    UNKNOWN_CLASS_LEADS = "UNKNOWN_CLASS_LEADS"
    PRIOR_COVERAGE_INCOMPLETE = "PRIOR_COVERAGE_INCOMPLETE"


def _bounded_probability(value: Any, *, field_name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a finite number in [0, 1]") from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be a finite number in [0, 1]; got {value!r}")
    return number


def normalize_hpo_id(value: Any) -> str:
    """Canonicalise ``hp:0000118``/``HP_0000118``/``HP0000118`` to ``HP:0000118``."""

    term = str(value).strip().upper().replace("_", ":")
    if term.startswith("HP") and ":" not in term and term[2:].isdigit():
        term = f"HP:{term[2:]}"
    if not HPO_ID_PATTERN.fullmatch(term):
        raise ValueError(f"Invalid HPO identifier: {value!r}")
    return term


@dataclass(frozen=True)
class SourceSpan:
    """Character offsets of the free-text mention an observation came from."""

    start: int
    end: int
    text: str = ""
    document_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid source span: [{self.start}, {self.end})")


@dataclass(frozen=True)
class PhenotypeObservation:
    """One HPO observation handed to Agent 2.

    ``hpo_id`` and ``status`` are the historical positional signature and stay
    first.  Everything after them is optional metadata from Agent 1; all
    defaults are deliberately neutral so that an unstated field never silently
    strengthens or weakens the evidence.

    ``extraction_confidence`` is how sure Agent 1 is that the phrase describes a
    real finding; ``linking_confidence`` is how sure it is that the phrase maps
    to *this* HPO term.  ``None`` means "not supplied", which is not the same as
    0.0 and is handled by :class:`StatusWeightPolicy`.
    """

    hpo_id: str
    status: PhenotypeStatus = PhenotypeStatus.PRESENT
    extraction_confidence: Optional[float] = None
    linking_confidence: Optional[float] = None
    reviewer_status: ReviewerStatus = ReviewerStatus.NOT_REVIEWED
    onset: Optional[str] = None
    gestational_age_weeks: Optional[float] = None
    subject: PhenotypeSubject = PhenotypeSubject.UNKNOWN
    source_spans: tuple[SourceSpan, ...] = ()
    mention_count: int = 1
    assertion_certainty: float = 1.0
    hpo_name: Optional[str] = None
    mention_text: str = ""
    observation_method: Optional[str] = None
    modifiers: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hpo_id", normalize_hpo_id(self.hpo_id))
        object.__setattr__(self, "status", PhenotypeStatus.coerce(self.status))
        object.__setattr__(self, "reviewer_status", ReviewerStatus.coerce(self.reviewer_status))
        object.__setattr__(self, "subject", PhenotypeSubject.coerce(self.subject))
        object.__setattr__(
            self,
            "extraction_confidence",
            _bounded_probability(self.extraction_confidence, field_name="extraction_confidence"),
        )
        object.__setattr__(
            self,
            "linking_confidence",
            _bounded_probability(self.linking_confidence, field_name="linking_confidence"),
        )
        if self.onset is not None:
            object.__setattr__(self, "onset", str(self.onset).strip() or None)
        if self.gestational_age_weeks is not None:
            weeks = float(self.gestational_age_weeks)
            if not math.isfinite(weeks) or weeks < 0.0 or weeks > 45.0:
                raise ValueError(f"gestational_age_weeks must be within [0, 45]; got {weeks!r}")
            object.__setattr__(self, "gestational_age_weeks", weeks)
        object.__setattr__(self, "source_spans", tuple(self.source_spans))
        count = int(self.mention_count)
        if count < 1:
            raise ValueError("mention_count must be at least 1")
        object.__setattr__(self, "mention_count", count)
        certainty = _bounded_probability(
            self.assertion_certainty, field_name="assertion_certainty"
        )
        object.__setattr__(self, "assertion_certainty", 1.0 if certainty is None else certainty)
        if self.hpo_name is not None:
            object.__setattr__(self, "hpo_name", str(self.hpo_name).strip() or None)
        object.__setattr__(self, "mention_text", str(self.mention_text or ""))
        if self.observation_method is not None:
            object.__setattr__(
                self, "observation_method", str(self.observation_method).strip() or None
            )
        object.__setattr__(self, "modifiers", dict(self.modifiers or {}))
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    @classmethod
    def coerce(cls, value: Any) -> "PhenotypeObservation":
        """Accept the historical shorthands as well as the full mapping form."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping):
            hpo_id = value.get("hpo_id", value.get("id"))
            if hpo_id is None:
                raise ValueError("Phenotype mapping requires 'hpo_id' or 'id'")
            return cls(
                str(hpo_id),
                PhenotypeStatus.coerce(value.get("status", value.get("assertion", "PRESENT"))),
                extraction_confidence=value.get("extraction_confidence"),
                linking_confidence=value.get("linking_confidence", value.get("model_confidence")),
                reviewer_status=ReviewerStatus.coerce(value.get("reviewer_status")),
                onset=value.get("onset"),
                gestational_age_weeks=value.get("gestational_age_weeks"),
                subject=PhenotypeSubject.coerce(value.get("subject")),
                source_spans=_coerce_spans(
                    value.get("source_spans") or value.get("source_span") or ()
                ),
                mention_count=int(value.get("mention_count", 1) or 1),
                assertion_certainty=value.get("assertion_certainty", value.get("certainty", 1.0)),
                hpo_name=value.get("hpo_name"),
                mention_text=value.get("mention_text", ""),
                observation_method=value.get("observation_method"),
                modifiers=dict(value.get("modifiers") or {}),
                provenance=dict(value.get("provenance") or {}),
            )
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            if len(value) == 1:
                return cls(str(value[0]))
            if len(value) == 2:
                return cls(str(value[0]), PhenotypeStatus.coerce(value[1]))
        raise TypeError(f"Cannot convert {type(value).__name__} to PhenotypeObservation")

    def merged_with(
        self, other: "PhenotypeObservation", status: PhenotypeStatus
    ) -> "PhenotypeObservation":
        """Combine two mentions of the same term without duplicating evidence.

        Provenance is unioned and ``mention_count`` accumulates, but the
        likelihood is later computed once from the resulting single observation.
        Confidence is taken as the maximum of the two: a second, more confident
        mention of the same finding is corroboration, not dilution.
        """

        def _best(left: Optional[float], right: Optional[float]) -> Optional[float]:
            values = [item for item in (left, right) if item is not None]
            return max(values) if values else None

        reviewer_rank = {
            ReviewerStatus.NOT_REVIEWED: 0,
            ReviewerStatus.CLINICIAN_MODIFIED: 1,
            ReviewerStatus.CLINICIAN_CONFIRMED: 2,
            ReviewerStatus.CLINICIAN_REJECTED: 3,
        }
        reviewer = max(
            (self.reviewer_status, other.reviewer_status), key=lambda item: reviewer_rank[item]
        )
        spans = tuple(dict.fromkeys((*self.source_spans, *other.source_spans)))
        return PhenotypeObservation(
            hpo_id=self.hpo_id,
            status=status,
            extraction_confidence=_best(self.extraction_confidence, other.extraction_confidence),
            linking_confidence=_best(self.linking_confidence, other.linking_confidence),
            reviewer_status=reviewer,
            onset=self.onset or other.onset,
            gestational_age_weeks=(
                self.gestational_age_weeks
                if self.gestational_age_weeks is not None
                else other.gestational_age_weeks
            ),
            subject=self.subject if self.subject != PhenotypeSubject.UNKNOWN else other.subject,
            source_spans=spans,
            mention_count=self.mention_count + other.mention_count,
            assertion_certainty=max(self.assertion_certainty, other.assertion_certainty),
            hpo_name=self.hpo_name or other.hpo_name,
            mention_text=self.mention_text or other.mention_text,
            observation_method=self.observation_method or other.observation_method,
            modifiers={**dict(other.modifiers), **dict(self.modifiers)},
            provenance={**dict(other.provenance), **dict(self.provenance)},
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hpo_id": self.hpo_id,
            "status": self.status.value,
            "extraction_confidence": self.extraction_confidence,
            "linking_confidence": self.linking_confidence,
            "reviewer_status": self.reviewer_status.value,
            "onset": self.onset,
            "gestational_age_weeks": self.gestational_age_weeks,
            "subject": self.subject.value,
            "source_spans": [
                {
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                    "document_id": span.document_id,
                }
                for span in self.source_spans
            ],
            "mention_count": self.mention_count,
            "assertion_certainty": self.assertion_certainty,
            "hpo_name": self.hpo_name,
            "mention_text": self.mention_text,
            "observation_method": self.observation_method,
            "modifiers": dict(self.modifiers),
            "provenance": dict(self.provenance),
        }


def _coerce_spans(value: Any) -> tuple[SourceSpan, ...]:
    if isinstance(value, (SourceSpan, Mapping)):
        value = [value]
    spans: list[SourceSpan] = []
    for item in value or ():
        if isinstance(item, SourceSpan):
            spans.append(item)
        elif isinstance(item, Mapping):
            spans.append(
                SourceSpan(
                    start=int(item.get("start", item.get("span_start", 0))),
                    end=int(item.get("end", item.get("span_end", 0))),
                    text=str(item.get("text", item.get("mention_text", "")) or ""),
                    document_id=item.get("document_id"),
                )
            )
        elif isinstance(item, Sequence) and not isinstance(item, str) and len(item) >= 2:
            spans.append(SourceSpan(int(item[0]), int(item[1])))
    return tuple(spans)


@dataclass(frozen=True)
class StatusWeightPolicy:
    """How assessment status and Agent 1 confidence become an evidence weight.

    The historical behaviour was a blind ``0.5`` for ``SUSPECTED``.  That value
    survives only as the *explicit fallback* used when no confidence at all was
    supplied; when Agent 1 does report confidence, the weight is derived from it
    and bounded, so one over-confident extraction cannot act like a confirmed
    finding.
    """

    present_weight: float = 1.0
    absent_weight: float = 1.0
    suspected_fallback_weight: float = 0.5
    suspected_max_weight: float = 0.75
    suspected_min_weight: float = 0.10
    #: Multiplier used when Agent 1 supplied no confidence at all.  ``1.0``
    #: keeps the historical behaviour: absence of a confidence score is not
    #: evidence of low confidence.
    missing_confidence_multiplier: float = 1.0
    #: A clinician-confirmed observation is trusted at full weight for its
    #: status regardless of what the model reported.
    reviewer_confirmed_overrides_confidence: bool = True

    def confidence_multiplier(self, observation: PhenotypeObservation) -> float:
        if (
            self.reviewer_confirmed_overrides_confidence
            and observation.reviewer_status == ReviewerStatus.CLINICIAN_CONFIRMED
        ):
            return 1.0
        values = [
            value
            for value in (observation.extraction_confidence, observation.linking_confidence)
            if value is not None
        ]
        if not values:
            return float(self.missing_confidence_multiplier)
        product = 1.0
        for value in values:
            product *= value
        return float(product)

    def weight_for(self, observation: PhenotypeObservation) -> float:
        """Return the evidence weight in ``[0, 1]`` for one observation."""

        if observation.reviewer_status == ReviewerStatus.CLINICIAN_REJECTED:
            return 0.0
        if observation.status == PhenotypeStatus.NOT_ASSESSED:
            return 0.0
        multiplier = self.confidence_multiplier(observation)
        if observation.status == PhenotypeStatus.PRESENT:
            return max(0.0, min(1.0, self.present_weight * multiplier))
        if observation.status == PhenotypeStatus.ABSENT:
            return max(0.0, min(1.0, self.absent_weight * multiplier))
        # SUSPECTED: use the documented fallback only when nothing better exists.
        if (
            observation.extraction_confidence is None
            and observation.linking_confidence is None
            and observation.reviewer_status != ReviewerStatus.CLINICIAN_CONFIRMED
        ):
            return float(self.suspected_fallback_weight)
        scaled = self.suspected_max_weight * multiplier
        return float(max(self.suspected_min_weight, min(self.suspected_max_weight, scaled)))


@dataclass(frozen=True)
class ClinicalContext:
    """Case-level context.  ``case_id`` must be a study code, not an identifier.

    No direct identifier (name, hospital record number, phone, address) belongs
    in this object.  A pattern check rejects the most common accidents; it is a
    guard-rail, not a de-identification service.
    """

    case_id: str
    fetal_sex: FetalSex = FetalSex.UNKNOWN
    gestational_age_weeks: Optional[float] = None
    maternal_age_years: Optional[float] = None
    maternal_age_permitted: bool = False
    family_history: tuple[str, ...] = ()
    referral_setting: Optional[str] = None
    cohort_id: Optional[str] = None
    hpo_release: Optional[str] = None

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        if not case_id:
            raise ValueError("case_id is required")
        if re.search(r"\d{9,}", case_id):
            raise ValueError(
                "case_id looks like a direct identifier (9+ consecutive digits); "
                "use a de-identified study code"
            )
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "fetal_sex", FetalSex.coerce(self.fetal_sex))
        for name in ("gestational_age_weeks", "maternal_age_years"):
            value = getattr(self, name)
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, number)
        if self.gestational_age_weeks is not None and self.gestational_age_weeks > 45.0:
            raise ValueError("gestational_age_weeks must be within [0, 45]")
        object.__setattr__(self, "family_history", tuple(self.family_history))

    @property
    def usable_maternal_age(self) -> Optional[float]:
        """Maternal age, but only when the study explicitly permitted its use."""

        return self.maternal_age_years if self.maternal_age_permitted else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "fetal_sex": self.fetal_sex.value,
            "gestational_age_weeks": self.gestational_age_weeks,
            "maternal_age_years": self.usable_maternal_age,
            "maternal_age_permitted": self.maternal_age_permitted,
            "family_history": list(self.family_history),
            "referral_setting": self.referral_setting,
            "cohort_id": self.cohort_id,
            "hpo_release": self.hpo_release,
        }


class Agent2ContractErrorCode(str, Enum):
    """Stable codes returned when the Agent-1 handoff is rejected."""

    INVALID_JSON = "INVALID_JSON"
    PAYLOAD_INTEGRITY_MISMATCH = "PAYLOAD_INTEGRITY_MISMATCH"
    UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"
    UNSUPPORTED_PAYLOAD_VERSION = "UNSUPPORTED_PAYLOAD_VERSION"
    SENSITIVE_FIELD = "SENSITIVE_FIELD"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_FIELD = "INVALID_FIELD"
    REVIEW_NOT_APPROVED = "REVIEW_NOT_APPROVED"
    NON_FETAL_SCORING_OBSERVATION = "NON_FETAL_SCORING_OBSERVATION"
    FETAL_CONTEXT_LEAKAGE = "FETAL_CONTEXT_LEAKAGE"
    NO_POSITIVE_FETAL_OBSERVATION = "NO_POSITIVE_FETAL_OBSERVATION"


class Agent2ContractError(ValueError):
    """Machine-readable, fail-closed Agent-1 to Agent-2 contract error."""

    def __init__(
        self,
        code: Agent2ContractErrorCode,
        message: str,
        *,
        path: str = "$",
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = Agent2ContractErrorCode(code)
        self.path = path
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": str(self),
            "path": self.path,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class Agent1ToAgent2Request:
    """Validated handoff. Only ``observations`` may enter disease scoring."""

    schema_version: str
    payload_version: str
    payload_sha256: str
    integrity_verified: bool
    case_id: str
    ontology_version: Optional[str]
    observations: tuple[PhenotypeObservation, ...]
    contextual_observations: tuple[PhenotypeObservation, ...]
    ignored_mentions: tuple[Mapping[str, Any], ...]
    context: ClinicalContext
    reviewer: str
    review_gate_version: str
    reviewed_profile_fingerprint: str
    report_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def clinician_approved(self) -> bool:
        # Construction is impossible until the strict review gate has passed.
        return True

    @property
    def review_status(self) -> str:
        return "approved"

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "payload_version": self.payload_version,
            "payload_sha256": self.payload_sha256,
            "integrity_verified": self.integrity_verified,
            "case_id": self.case_id,
            "ontology_version": self.ontology_version,
            "report_ids": list(self.report_ids),
            "review_status": self.review_status,
            "clinician_approved": self.clinician_approved,
            "reviewer": self.reviewer,
            "review_gate_version": self.review_gate_version,
            "reviewed_profile_fingerprint": self.reviewed_profile_fingerprint,
            "contextual_observation_count": len(self.contextual_observations),
            "ignored_mention_count": len(self.ignored_mentions),
        }


_SENSITIVE_CONTRACT_KEYS = {
    "source_text",
    "raw_text",
    "report_text",
    "clinical_note",
    "note_text",
    "patient_name",
    "patient_full_name",
    "medical_record_number",
    "mrn",
    "phone",
    "address",
}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "payload_version",
    "case_id",
    "ontology_version",
    "observations",
    "contextual_observations",
    "ignored_mentions",
    "clinical_context",
    "review",
    "confidence_semantics",
    "warnings",
    "research_use_only",
    "diagnostic_claim",
}

_OBSERVATION_KEYS = {
    "hpo_id",
    "hpo_name",
    "status",
    "mention_text",
    "source_span",
    "subject",
    "assertion_certainty",
    "extraction_confidence",
    "linking_confidence",
    "reviewer_status",
    "gestational_age_weeks",
    "observation_method",
    "modifiers",
    "provenance",
}


def _contract_error(
    code: Agent2ContractErrorCode,
    message: str,
    path: str,
    **details: Any,
) -> Agent2ContractError:
    return Agent2ContractError(code, message, path=path, details=details)


def _reject_sensitive_contract_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            nested_path = f"{path}.{key_text}"
            if key_text.casefold() in _SENSITIVE_CONTRACT_KEYS:
                raise _contract_error(
                    Agent2ContractErrorCode.SENSITIVE_FIELD,
                    f"Sensitive/raw-text field {key_text!r} is forbidden in the Agent-2 handoff.",
                    nested_path,
                    field=key_text,
                )
            _reject_sensitive_contract_keys(nested, nested_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_sensitive_contract_keys(nested, f"{path}[{index}]")


def _strict_mapping(
    value: Any,
    *,
    path: str,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "Expected an object.",
            path,
            actual_type=type(value).__name__,
        )
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise _contract_error(
            Agent2ContractErrorCode.UNKNOWN_FIELD,
            f"Unknown contract field(s): {', '.join(unknown)}.",
            path,
            fields=unknown,
        )
    missing = sorted(key for key in required if key not in value)
    if missing:
        raise _contract_error(
            Agent2ContractErrorCode.MISSING_FIELD,
            f"Missing required contract field(s): {', '.join(missing)}.",
            path,
            fields=missing,
        )
    return value


def _contract_string(value: Any, *, path: str, nullable: bool = False) -> Optional[str]:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "Expected a non-empty string.",
            path,
        )
    return value.strip()


def _contract_number(
    value: Any,
    *,
    path: str,
    minimum: float,
    maximum: float,
    nullable: bool = False,
) -> Optional[float]:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise _contract_error(Agent2ContractErrorCode.INVALID_FIELD, "Expected a number.", path)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD, "Expected a finite number.", path
        ) from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            f"Expected a finite number within [{minimum}, {maximum}].",
            path,
        )
    return number


def _load_contract_payload(
    payload: Mapping[str, Any] | str | bytes | bytearray,
) -> tuple[Mapping[str, Any], bytes]:
    if isinstance(payload, Mapping):
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as exc:
            raise _contract_error(
                Agent2ContractErrorCode.INVALID_JSON,
                f"Payload is not strict JSON: {exc}",
                "$",
            ) from exc
    elif isinstance(payload, str):
        encoded = payload.encode("utf-8")
    elif isinstance(payload, (bytes, bytearray)):
        encoded = bytes(payload)
    else:
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_JSON,
            "Payload must be an object, UTF-8 JSON string, or JSON bytes.",
            "$",
        )

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _contract_error(
                    Agent2ContractErrorCode.INVALID_JSON,
                    f"Duplicate JSON key {key!r} is forbidden.",
                    "$",
                    duplicate_key=key,
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_JSON,
            f"Non-finite JSON number {value!r} is forbidden.",
            "$",
        )

    try:
        decoded = encoded.decode("utf-8")
        parsed = json.loads(
            decoded,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except Agent2ContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_JSON,
            f"Payload is not one strict UTF-8 JSON document: {exc}",
            "$",
        ) from exc
    if not isinstance(parsed, Mapping):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_JSON, "Top-level JSON value must be an object.", "$"
        )
    return parsed, encoded


def _parse_source_span(
    value: Any, *, path: str, mention_text: str, report_id: Optional[str]
) -> tuple[SourceSpan, ...]:
    span = _strict_mapping(
        value,
        path=path,
        allowed={"start", "end"},
        required={"start", "end"},
    )
    start = span["start"]
    end = span["end"]
    if isinstance(start, bool) or not isinstance(start, int):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD, "Span start must be an integer.", f"{path}.start"
        )
    if isinstance(end, bool) or not isinstance(end, int):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD, "Span end must be an integer.", f"{path}.end"
        )
    try:
        return (SourceSpan(start, end, text=mention_text, document_id=report_id),)
    except ValueError as exc:
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD, str(exc), path
        ) from exc


def _parse_agent1_observation(
    value: Any, *, path: str, scoring: bool
) -> PhenotypeObservation:
    item = _strict_mapping(
        value,
        path=path,
        allowed=_OBSERVATION_KEYS,
        required={
            "hpo_id",
            "status",
            "subject",
            "assertion_certainty",
            "reviewer_status",
            "source_span",
            "modifiers",
            "provenance",
        },
    )
    subject = PhenotypeSubject.coerce(item["subject"])
    if scoring and subject is not PhenotypeSubject.FETUS:
        raise _contract_error(
            Agent2ContractErrorCode.NON_FETAL_SCORING_OBSERVATION,
            "Only fetal observations may enter disease scoring.",
            f"{path}.subject",
            subject=subject.value,
        )
    if not scoring and subject is PhenotypeSubject.FETUS:
        raise _contract_error(
            Agent2ContractErrorCode.FETAL_CONTEXT_LEAKAGE,
            "A fetal observation was placed in contextual_observations.",
            f"{path}.subject",
        )
    if item["reviewer_status"] != "approved":
        raise _contract_error(
            Agent2ContractErrorCode.REVIEW_NOT_APPROVED,
            "Every observation must have reviewer_status='approved'.",
            f"{path}.reviewer_status",
        )

    modifiers = _strict_mapping(
        item["modifiers"],
        path=f"{path}.modifiers",
        allowed={"laterality", "severity", "distribution", "progression", "measurements"},
    )
    measurements = modifiers.get("measurements", {})
    if not isinstance(measurements, Mapping):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "modifiers.measurements must be an object.",
            f"{path}.modifiers.measurements",
        )

    provenance = _strict_mapping(
        item["provenance"],
        path=f"{path}.provenance",
        allowed={"report_id", "extractor", "matched_alias", "clinician_action"},
    )
    report_id = _contract_string(
        provenance.get("report_id"), path=f"{path}.provenance.report_id", nullable=True
    )
    mention_text = item.get("mention_text", "")
    if not isinstance(mention_text, str):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "mention_text must be a string.",
            f"{path}.mention_text",
        )
    spans = _parse_source_span(
        item["source_span"],
        path=f"{path}.source_span",
        mention_text=mention_text,
        report_id=report_id,
    )
    try:
        return PhenotypeObservation(
            hpo_id=str(item["hpo_id"]),
            status=PhenotypeStatus.coerce(item["status"]),
            extraction_confidence=_contract_number(
                item.get("extraction_confidence"),
                path=f"{path}.extraction_confidence",
                minimum=0.0,
                maximum=1.0,
                nullable=True,
            ),
            linking_confidence=_contract_number(
                item.get("linking_confidence"),
                path=f"{path}.linking_confidence",
                minimum=0.0,
                maximum=1.0,
                nullable=True,
            ),
            reviewer_status=ReviewerStatus.CLINICIAN_CONFIRMED,
            gestational_age_weeks=_contract_number(
                item.get("gestational_age_weeks"),
                path=f"{path}.gestational_age_weeks",
                minimum=4.0,
                maximum=45.0,
                nullable=True,
            ),
            subject=subject,
            source_spans=spans,
            assertion_certainty=float(
                _contract_number(
                    item["assertion_certainty"],
                    path=f"{path}.assertion_certainty",
                    minimum=0.0,
                    maximum=1.0,
                )
            ),
            hpo_name=(
                _contract_string(item.get("hpo_name"), path=f"{path}.hpo_name", nullable=True)
            ),
            mention_text=mention_text,
            observation_method=_contract_string(
                item.get("observation_method"),
                path=f"{path}.observation_method",
                nullable=True,
            ),
            modifiers=dict(modifiers),
            provenance=dict(provenance),
        )
    except Agent2ContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD, str(exc), path
        ) from exc


def _parse_clinical_context(
    value: Any, *, case_id: str, ontology_version: Optional[str]
) -> ClinicalContext:
    path = "$.clinical_context"
    item = _strict_mapping(
        value,
        path=path,
        allowed={
            "gestational_age_weeks",
            "fetal_sex",
            "maternal_age_years",
            "maternal_age_permitted",
            "family_history",
            "referral_setting",
            "cohort_id",
            "hpo_release",
        },
    )
    supplied_release = _contract_string(
        item.get("hpo_release"), path=f"{path}.hpo_release", nullable=True
    )
    if supplied_release and ontology_version and supplied_release != ontology_version:
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "clinical_context.hpo_release conflicts with ontology_version.",
            f"{path}.hpo_release",
            ontology_version=ontology_version,
        )
    history = item.get("family_history", ())
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray)):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "family_history must be an array of strings.",
            f"{path}.family_history",
        )
    if any(not isinstance(entry, str) for entry in history):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "family_history must contain only strings.",
            f"{path}.family_history",
        )
    permitted = item.get("maternal_age_permitted", False)
    if not isinstance(permitted, bool):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "maternal_age_permitted must be boolean.",
            f"{path}.maternal_age_permitted",
        )
    try:
        return ClinicalContext(
            case_id=case_id,
            fetal_sex=FetalSex.coerce(item.get("fetal_sex")),
            gestational_age_weeks=_contract_number(
                item.get("gestational_age_weeks"),
                path=f"{path}.gestational_age_weeks",
                minimum=0.0,
                maximum=45.0,
                nullable=True,
            ),
            maternal_age_years=_contract_number(
                item.get("maternal_age_years"),
                path=f"{path}.maternal_age_years",
                minimum=0.0,
                maximum=120.0,
                nullable=True,
            ),
            maternal_age_permitted=permitted,
            family_history=tuple(history),
            referral_setting=_contract_string(
                item.get("referral_setting"), path=f"{path}.referral_setting", nullable=True
            ),
            cohort_id=_contract_string(
                item.get("cohort_id"), path=f"{path}.cohort_id", nullable=True
            ),
            hpo_release=supplied_release or ontology_version,
        )
    except Agent2ContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise _contract_error(Agent2ContractErrorCode.INVALID_FIELD, str(exc), path) from exc


def parse_agent1_to_agent2_request(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    expected_payload_sha256: Optional[str] = None,
) -> Agent1ToAgent2Request:
    """Strictly adapt the reviewed Agent-1 payload without importing Agent 1.

    ``profile_to_agent2_request`` currently encodes its payload version in
    ``schema_version`` and omits the redundant ``payload_version`` key.  This
    adapter derives ``1.0`` for that canonical form, but if a sender supplies
    ``payload_version`` it must match exactly.  Byte-level integrity is checked
    whenever the trusted caller supplies ``expected_payload_sha256``.
    """

    raw, encoded = _load_contract_payload(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    if expected_payload_sha256 is not None:
        expected = str(expected_payload_sha256).strip().casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise _contract_error(
                Agent2ContractErrorCode.INVALID_FIELD,
                "expected_payload_sha256 must be 64 lowercase/uppercase hexadecimal characters.",
                "$",
            )
        if not hmac.compare_digest(digest, expected):
            raise _contract_error(
                Agent2ContractErrorCode.PAYLOAD_INTEGRITY_MISMATCH,
                "Agent-1 payload bytes do not match the trusted SHA-256 digest.",
                "$",
                expected_sha256=expected,
                actual_sha256=digest,
            )

    _reject_sensitive_contract_keys(raw)
    item = _strict_mapping(
        raw,
        path="$",
        allowed=_TOP_LEVEL_KEYS,
        required={
            "schema_version",
            "case_id",
            "observations",
            "contextual_observations",
            "ignored_mentions",
            "clinical_context",
            "review",
            "confidence_semantics",
            "warnings",
            "research_use_only",
            "diagnostic_claim",
        },
    )
    if item["schema_version"] != AGENT1_TO_AGENT2_SCHEMA_VERSION:
        raise _contract_error(
            Agent2ContractErrorCode.UNSUPPORTED_SCHEMA_VERSION,
            f"Expected schema_version={AGENT1_TO_AGENT2_SCHEMA_VERSION!r}.",
            "$.schema_version",
            actual=item["schema_version"],
        )
    declared_payload_version = item.get("payload_version", AGENT1_TO_AGENT2_PAYLOAD_VERSION)
    if declared_payload_version != AGENT1_TO_AGENT2_PAYLOAD_VERSION:
        raise _contract_error(
            Agent2ContractErrorCode.UNSUPPORTED_PAYLOAD_VERSION,
            f"Expected payload_version={AGENT1_TO_AGENT2_PAYLOAD_VERSION!r}.",
            "$.payload_version",
            actual=declared_payload_version,
        )
    if item["research_use_only"] is not True or item["diagnostic_claim"] is not False:
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "The handoff must declare research_use_only=true and diagnostic_claim=false.",
            "$",
        )

    case_id = str(_contract_string(item["case_id"], path="$.case_id"))
    ontology_version = _contract_string(
        item.get("ontology_version"), path="$.ontology_version", nullable=True
    )

    review = _strict_mapping(
        item["review"],
        path="$.review",
        allowed={"approved", "reviewer", "review_gate_version", "reviewed_profile_fingerprint"},
        required={"approved", "reviewer", "review_gate_version", "reviewed_profile_fingerprint"},
    )
    if review["approved"] is not True:
        raise _contract_error(
            Agent2ContractErrorCode.REVIEW_NOT_APPROVED,
            "Agent 2 accepts only a clinician-approved review gate.",
            "$.review.approved",
        )
    reviewer = str(_contract_string(review["reviewer"], path="$.review.reviewer"))
    review_gate_version = str(
        _contract_string(review["review_gate_version"], path="$.review.review_gate_version")
    )
    fingerprint = str(
        _contract_string(
            review["reviewed_profile_fingerprint"],
            path="$.review.reviewed_profile_fingerprint",
        )
    )
    if not re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "reviewed_profile_fingerprint must be a SHA-256 hex digest.",
            "$.review.reviewed_profile_fingerprint",
        )

    raw_observations = item["observations"]
    raw_contextual = item["contextual_observations"]
    if not isinstance(raw_observations, list) or not isinstance(raw_contextual, list):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "observations and contextual_observations must be arrays.",
            "$",
        )
    observations = tuple(
        _parse_agent1_observation(value, path=f"$.observations[{index}]", scoring=True)
        for index, value in enumerate(raw_observations)
    )
    contextual = tuple(
        _parse_agent1_observation(
            value, path=f"$.contextual_observations[{index}]", scoring=False
        )
        for index, value in enumerate(raw_contextual)
    )
    if not any(
        observation.status in {PhenotypeStatus.PRESENT, PhenotypeStatus.SUSPECTED}
        for observation in observations
    ):
        raise _contract_error(
            Agent2ContractErrorCode.NO_POSITIVE_FETAL_OBSERVATION,
            "At least one approved fetal PRESENT or SUSPECTED HPO is required.",
            "$.observations",
        )

    raw_ignored = item["ignored_mentions"]
    if not isinstance(raw_ignored, list):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "ignored_mentions must be an array.",
            "$.ignored_mentions",
        )
    ignored: list[Mapping[str, Any]] = []
    for index, value in enumerate(raw_ignored):
        ignored_item = _strict_mapping(
            value,
            path=f"$.ignored_mentions[{index}]",
            allowed={"mention_text", "reason", "source_span"},
            required={"mention_text", "reason", "source_span"},
        )
        mention = ignored_item["mention_text"]
        if not isinstance(mention, str):
            raise _contract_error(
                Agent2ContractErrorCode.INVALID_FIELD,
                "mention_text must be a string.",
                f"$.ignored_mentions[{index}].mention_text",
            )
        _parse_source_span(
            ignored_item["source_span"],
            path=f"$.ignored_mentions[{index}].source_span",
            mention_text=mention,
            report_id=None,
        )
        _contract_string(
            ignored_item["reason"], path=f"$.ignored_mentions[{index}].reason"
        )
        ignored.append(dict(ignored_item))

    semantics = _strict_mapping(
        item["confidence_semantics"],
        path="$.confidence_semantics",
        allowed={
            "assertion_certainty",
            "extraction_confidence",
            "linking_confidence",
            "reviewer_status",
        },
        required={
            "assertion_certainty",
            "extraction_confidence",
            "linking_confidence",
            "reviewer_status",
        },
    )
    for key, value in semantics.items():
        _contract_string(value, path=f"$.confidence_semantics.{key}")

    warnings = item["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(value, str) for value in warnings):
        raise _contract_error(
            Agent2ContractErrorCode.INVALID_FIELD,
            "warnings must be an array of strings.",
            "$.warnings",
        )
    context = _parse_clinical_context(
        item["clinical_context"], case_id=case_id, ontology_version=ontology_version
    )
    report_ids = tuple(
        sorted(
            {
                str(observation.provenance["report_id"])
                for observation in (*observations, *contextual)
                if observation.provenance.get("report_id")
            }
        )
    )
    return Agent1ToAgent2Request(
        schema_version=AGENT1_TO_AGENT2_SCHEMA_VERSION,
        payload_version=AGENT1_TO_AGENT2_PAYLOAD_VERSION,
        payload_sha256=digest,
        integrity_verified=expected_payload_sha256 is not None,
        case_id=case_id,
        ontology_version=ontology_version,
        observations=observations,
        contextual_observations=contextual,
        ignored_mentions=tuple(ignored),
        context=context,
        reviewer=reviewer,
        review_gate_version=review_gate_version,
        reviewed_profile_fingerprint=fingerprint.casefold(),
        report_ids=report_ids,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class DiseasePrior:
    """P(disease) before any phenotype evidence, with mandatory provenance."""

    disease_id: str
    prior_probability: float
    population: str
    source: str
    version: Optional[str] = None
    cohort_id: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        disease_id = str(self.disease_id).strip()
        if not disease_id:
            raise ValueError("DiseasePrior.disease_id is required")
        object.__setattr__(self, "disease_id", disease_id)
        try:
            probability = float(self.prior_probability)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("prior_probability must be a finite number in (0, 1)") from exc
        if not math.isfinite(probability) or probability <= 0.0 or probability >= 1.0:
            raise ValueError(
                f"prior_probability must satisfy 0 < p < 1; got {self.prior_probability!r}"
            )
        object.__setattr__(self, "prior_probability", probability)
        if not str(self.population).strip():
            raise ValueError("DiseasePrior.population must name the population the prior describes")
        if not str(self.source).strip():
            raise ValueError("DiseasePrior.source must cite where the prior came from")
        object.__setattr__(self, "population", str(self.population).strip())
        object.__setattr__(self, "source", str(self.source).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "prior_probability": self.prior_probability,
            "population": self.population,
            "source": self.source,
            "version": self.version,
            "cohort_id": self.cohort_id,
            "notes": self.notes,
        }


class DiseasePriorProvider(Protocol):
    """Interface for supplying priors; no prevalence is ever invented here."""

    def prior_for(
        self, disease_id: str, context: Optional[ClinicalContext] = None
    ) -> Optional[DiseasePrior]:
        ...


@dataclass
class StaticDiseasePriorProvider:
    """Prior lookup backed by an explicit, curated table.

    A cohort-specific entry wins over a sex-specific one, which wins over the
    general table.  A missing entry returns ``None`` rather than a guess, which
    forces the caller into ranking-only mode.
    """

    priors: Mapping[str, DiseasePrior] = field(default_factory=dict)
    by_cohort: Mapping[str, Mapping[str, DiseasePrior]] = field(default_factory=dict)
    sex_specific: Mapping[str, Mapping[str, DiseasePrior]] = field(default_factory=dict)

    def prior_for(
        self, disease_id: str, context: Optional[ClinicalContext] = None
    ) -> Optional[DiseasePrior]:
        key = str(disease_id)
        if context is not None:
            if context.cohort_id:
                cohort_table = self.by_cohort.get(context.cohort_id) or {}
                if key in cohort_table:
                    return cohort_table[key]
            if context.fetal_sex != FetalSex.UNKNOWN:
                sex_table = self.sex_specific.get(context.fetal_sex.value) or {}
                if key in sex_table:
                    return sex_table[key]
        return self.priors.get(key)


@dataclass(frozen=True)
class CalibratorMetadata:
    """Everything needed to decide whether a calibrator may be trusted.

    ``clinical_ready`` is never asserted by this package on its own.  It becomes
    true only after a fitted calibrator has been evaluated on real,
    de-identified, clinician-confirmed cases split by patient/family with no
    leakage, and a human records that judgement in this metadata.
    """

    method: str
    feature_names: tuple[str, ...]
    feature_version: str
    hpo_release: Optional[str]
    disease_universe: str
    training_cohort: str
    validation_cohort: Optional[str] = None
    test_cohort: Optional[str] = None
    split_unit: str = "patient"
    synthetic_only: bool = True
    clinician_reviewed: bool = False
    negative_sampling: Optional[str] = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    fitted_at: Optional[str] = None
    schema_version: str = "agent2-calibrator/1"

    def __post_init__(self) -> None:
        if not str(self.method).strip():
            raise ValueError("CalibratorMetadata.method is required")
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        if not self.feature_names:
            raise ValueError("CalibratorMetadata.feature_names must not be empty")
        if not str(self.training_cohort).strip():
            raise ValueError("CalibratorMetadata.training_cohort must name the fitting cohort")
        if not str(self.disease_universe).strip():
            raise ValueError("CalibratorMetadata.disease_universe is required")
        if self.split_unit not in {"patient", "family", "case"}:
            raise ValueError("split_unit must be 'patient', 'family' or 'case'")

    @property
    def clinical_ready(self) -> bool:
        """True only for a non-synthetic, reviewed calibrator with a held-out split."""

        return bool(
            not self.synthetic_only
            and self.clinician_reviewed
            and self.validation_cohort
            and self.test_cohort
            and self.split_unit in {"patient", "family"}
        )

    def incompatibility_with(
        self, *, feature_names: Sequence[str], hpo_release: Optional[str]
    ) -> Optional[str]:
        """Return a human-readable reason the calibrator must not be used, if any."""

        if tuple(feature_names) != self.feature_names:
            return (
                f"calibrator feature vector {list(self.feature_names)} does not match "
                f"the current {list(feature_names)}"
            )
        if self.hpo_release and hpo_release and self.hpo_release != hpo_release:
            return (
                f"calibrator was fitted on HPO release {self.hpo_release!r} but the loaded "
                f"snapshot is {hpo_release!r}"
            )
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "feature_names": list(self.feature_names),
            "feature_version": self.feature_version,
            "hpo_release": self.hpo_release,
            "disease_universe": self.disease_universe,
            "training_cohort": self.training_cohort,
            "validation_cohort": self.validation_cohort,
            "test_cohort": self.test_cohort,
            "split_unit": self.split_unit,
            "synthetic_only": self.synthetic_only,
            "clinician_reviewed": self.clinician_reviewed,
            "clinical_ready": self.clinical_ready,
            "negative_sampling": self.negative_sampling,
            "metrics": dict(self.metrics),
            "fitted_at": self.fitted_at,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class Agent2Warning:
    code: WarningCode
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "details": dict(self.details)}


@dataclass(frozen=True)
class Abstention:
    """A structured refusal to conclude."""

    reason: AbstentionReason
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason.value, "message": self.message, "details": dict(self.details)}


class DiseaseIdentityResolver(Protocol):
    """Maps OMIM/ORPHA/MONDO IDs onto one concept using an external mapping table.

    Name similarity is never sufficient.  Implementations must be backed by a
    curated cross-reference file; when none is loaded, the identity function is
    used and no disease is dropped or merged.
    """

    def resolve(self, disease_id: str) -> str:
        ...

    def label(self, concept_id: str) -> Optional[str]:
        ...


@dataclass
class IdentityDiseaseResolver:
    """Default resolver: keeps raw IDs, merges nothing, drops nothing."""

    mapping: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    source: Optional[str] = None

    def resolve(self, disease_id: str) -> str:
        return self.mapping.get(str(disease_id), str(disease_id))

    def label(self, concept_id: str) -> Optional[str]:
        return self.labels.get(str(concept_id))

    @property
    def is_identity(self) -> bool:
        return not self.mapping


def load_disease_xref(path: str | Path) -> IdentityDiseaseResolver:
    """Resolver from a curated table with ``concept_id``, ``omim_id`` and ``orpha_id`` columns.

    Each row declares one OMIM and one ORPHA ID as the same disease.  A member ID
    listed under two concepts is a curation error and is rejected rather than
    resolved arbitrarily.
    """

    path = Path(path)
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t")
    required = {"concept_id", "omim_id", "orpha_id"}
    if not reader.fieldnames or not required <= set(reader.fieldnames):
        raise ValueError(f"{path.name}: expected columns {sorted(required)}, got {reader.fieldnames}")
    mapping: dict[str, str] = {}
    labels: dict[str, str] = {}
    for line, row in enumerate(reader, start=2):
        concept = (row["concept_id"] or "").strip()
        members = ((row["omim_id"] or "").strip(), (row["orpha_id"] or "").strip())
        if not concept or not members[0].startswith("OMIM:") or not members[1].startswith("ORPHA:"):
            raise ValueError(f"{path.name}:{line}: malformed cross-reference row")
        for member in members:
            if mapping.get(member, concept) != concept:
                raise ValueError(f"{path.name}:{line}: {member} is mapped to {mapping[member]} and {concept}")
            mapping[member] = concept
        name = (row.get("omim_name") or row.get("orpha_name") or "").strip()
        if name:
            labels.setdefault(concept, name)
    if not mapping:
        raise ValueError(f"{path.name}: no cross-reference rows")
    return IdentityDiseaseResolver(
        mapping=mapping,
        labels=labels,
        source=f"{path.name} sha256={hashlib.sha256(raw).hexdigest()}",
    )


__all__ = [
    "AGENT1_TO_AGENT2_PAYLOAD_VERSION",
    "AGENT1_TO_AGENT2_SCHEMA_VERSION",
    "PHENOTYPIC_ABNORMALITY_ROOT",
    "Agent1ToAgent2Request",
    "Agent2ContractError",
    "Agent2ContractErrorCode",
    "Abstention",
    "AbstentionReason",
    "Agent2Warning",
    "BackgroundModel",
    "CalibratorMetadata",
    "ClinicalContext",
    "DiseaseIdentityResolver",
    "DiseasePrior",
    "DiseasePriorProvider",
    "FetalSex",
    "IdentityDiseaseResolver",
    "MissingFrequencyPolicy",
    "PhenotypeObservation",
    "PhenotypeStatus",
    "PhenotypeSubject",
    "ProbabilityKind",
    "ProbabilityScope",
    "RankingStrategy",
    "ReviewerStatus",
    "SourceSpan",
    "StaticDiseasePriorProvider",
    "StatusWeightPolicy",
    "WarningCode",
    "load_disease_xref",
    "normalize_hpo_id",
    "parse_agent1_to_agent2_request",
]
