"""Fail-closed intake, training and artifact handling for Agent-2 calibration.

This module ships no patient data and does not make a calibrator clinically
ready by itself.  Synthetic data are useful only for exercising the workflow.
Real-mode training requires reviewed, patient-level labels, leakage-safe splits
and an explicit caller-supplied approval attestation bound to the source hash.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .agent2_matcher import FEATURE_VERSION, HPOAgent2Matcher
from .agent2_schema import (
    CalibratorMetadata,
    PhenotypeObservation,
    PhenotypeStatus,
    PhenotypeSubject,
    ReviewerStatus,
)


CALIBRATION_CASE_SCHEMA_VERSION = "agent2-calibration-cases/1"
CALIBRATOR_ARTIFACT_VERSION = "agent2-calibrator-artifact/1"
DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "agent2_calibration_cases.schema.json"
)
SPLITS = ("train", "validation", "test")
REFERENCE_STANDARD_TYPES = {
    "molecular_confirmation",
    "multidisciplinary_consensus",
    "validated_registry_label",
}


class CalibrationErrorCode(str, Enum):
    INVALID_JSONL = "INVALID_JSONL"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_FIELD = "INVALID_FIELD"
    DIRECT_IDENTIFIER_RISK = "DIRECT_IDENTIFIER_RISK"
    DUPLICATE_CASE = "DUPLICATE_CASE"
    DUPLICATE_PATIENT = "DUPLICATE_PATIENT"
    SPLIT_LEAKAGE = "SPLIT_LEAKAGE"
    DUPLICATE_HPO = "DUPLICATE_HPO"
    CONTRADICTORY_HPO = "CONTRADICTORY_HPO"
    NO_POSITIVE_FINDING = "NO_POSITIVE_FINDING"
    UNKNOWN_HPO = "UNKNOWN_HPO"
    UNKNOWN_GOLD_DISEASE = "UNKNOWN_GOLD_DISEASE"
    REVIEW_NOT_APPROVED = "REVIEW_NOT_APPROVED"
    DATASET_MODE_MISMATCH = "DATASET_MODE_MISMATCH"
    SOURCE_INTEGRITY_MISMATCH = "SOURCE_INTEGRITY_MISMATCH"
    HPO_RELEASE_MISMATCH = "HPO_RELEASE_MISMATCH"
    SPLIT_MISSING = "SPLIT_MISSING"
    CLASS_MISSING = "CLASS_MISSING"
    ATTESTATION_REQUIRED = "ATTESTATION_REQUIRED"
    ATTESTATION_INVALID = "ATTESTATION_INVALID"
    ARTIFACT_EXISTS = "ARTIFACT_EXISTS"
    ARTIFACT_MISMATCH = "ARTIFACT_MISMATCH"


class CalibrationDataError(ValueError):
    """Validation failure with a stable code and JSON path."""

    def __init__(
        self,
        code: CalibrationErrorCode,
        message: str,
        *,
        path: str = "$",
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = CalibrationErrorCode(code)
        self.path = path
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": str(self),
            "path": self.path,
            "details": dict(self.details),
        }


class CalibrationArtifactError(CalibrationDataError):
    pass


@dataclass(frozen=True)
class ReferenceStandard:
    type: str
    source: str
    confirmed_by: str
    confirmed_at: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "source": self.source,
            "confirmed_by": self.confirmed_by,
            "confirmed_at": self.confirmed_at,
        }


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    patient_id: str
    family_id: str
    group_id: str
    split: str
    observations: tuple[PhenotypeObservation, ...]
    gold_disease_ids: tuple[str, ...]
    reference_standard: ReferenceStandard
    review_status: str
    synthetic: bool
    cohort_id: str
    hpo_release: str

    def digest_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "family_id": self.family_id,
            "group_id": self.group_id,
            "split": self.split,
            "observations": [item.as_dict() for item in self.observations],
            "gold_disease_ids": list(self.gold_disease_ids),
            "reference_standard": self.reference_standard.as_dict(),
            "review_status": self.review_status,
            "synthetic": self.synthetic,
            "cohort_id": self.cohort_id,
            "hpo_release": self.hpo_release,
        }


@dataclass(frozen=True)
class ValidatedCalibrationDataset:
    cases: tuple[CalibrationCase, ...]
    source_path: Path
    source_sha256: str
    real_mode: bool
    matcher_hpo_release: Optional[str]

    @property
    def by_split(self) -> dict[str, tuple[CalibrationCase, ...]]:
        return {
            split: tuple(case for case in self.cases if case.split == split)
            for split in SPLITS
        }

    @property
    def split_sha256(self) -> Mapping[str, str]:
        return {
            split: _sha256_json([case.digest_dict() for case in self.by_split[split]])
            for split in SPLITS
        }

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_CASE_SCHEMA_VERSION,
            "source": str(self.source_path),
            "source_sha256": self.source_sha256,
            "real_mode": self.real_mode,
            "clinical_ready": False,
            "case_count": len(self.cases),
            "split_case_counts": {
                split: len(cases) for split, cases in self.by_split.items()
            },
            "cohort_ids": sorted({case.cohort_id for case in self.cases}),
            "hpo_releases": sorted({case.hpo_release for case in self.cases}),
            "synthetic_case_count": sum(case.synthetic for case in self.cases),
            "split_sha256": dict(self.split_sha256),
        }


@dataclass(frozen=True)
class CandidateFeatureRow:
    case_id: str
    patient_id: str
    family_id: str
    group_id: str
    split: str
    disease_id: str
    label: int
    features: tuple[float, ...]
    sample_weight: float
    selection_reason: str

    def digest_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "family_id": self.family_id,
            "group_id": self.group_id,
            "split": self.split,
            "disease_id": self.disease_id,
            "label": self.label,
            "features": list(self.features),
            "sample_weight": self.sample_weight,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class CandidateFeatureDataset:
    rows: tuple[CandidateFeatureRow, ...]
    feature_names: tuple[str, ...]
    feature_version: str
    random_state: int
    train_negative_count: int
    negative_sampling_description: str

    @property
    def by_split(self) -> dict[str, tuple[CandidateFeatureRow, ...]]:
        return {
            split: tuple(row for row in self.rows if row.split == split) for split in SPLITS
        }

    @property
    def sha256(self) -> str:
        return _sha256_json([row.digest_dict() for row in self.rows])


@dataclass(frozen=True)
class CalibrationTrainingResult:
    model: "DeterministicLogisticCalibrator"
    metadata: CalibratorMetadata
    threshold: float
    validation_metrics: Mapping[str, Optional[float]]
    test_metrics: Mapping[str, Optional[float]]
    feature_dataset: CandidateFeatureDataset
    approval_attestation_sha256: Optional[str]
    approval_id: Optional[str]
    test_evaluation_count: int = 1

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.metadata.method,
            "clinical_ready": self.metadata.clinical_ready,
            "synthetic_only": self.metadata.synthetic_only,
            "threshold_selected_on": "validation",
            "threshold": self.threshold,
            "validation_metrics": dict(self.validation_metrics),
            "test_metrics": dict(self.test_metrics),
            "test_evaluation_count": self.test_evaluation_count,
            "feature_rows_sha256": self.feature_dataset.sha256,
        }


@dataclass
class DeterministicLogisticCalibrator:
    """Small deterministic weighted logistic regression with JSON-safe state."""

    mean_: list[float] = field(default_factory=list)
    scale_: list[float] = field(default_factory=list)
    coef_: list[float] = field(default_factory=list)
    intercept_: float = 0.0
    iterations: int = 2000
    learning_rate: float = 0.05
    l2: float = 1e-3

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        sample_weights: Sequence[float],
    ) -> "DeterministicLogisticCalibrator":
        if not rows or len(rows) != len(labels) or len(rows) != len(sample_weights):
            raise ValueError("Training rows, labels and sample weights must be non-empty and aligned")
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise ValueError("Training feature rows must have one consistent non-zero width")
        matrix = [[float(value) for value in row] for row in rows]
        if not all(math.isfinite(value) for row in matrix for value in row):
            raise ValueError("Training features must be finite")
        target = [int(value) for value in labels]
        if set(target) != {0, 1}:
            raise ValueError("Training labels must contain both binary classes")
        weights = [float(value) for value in sample_weights]
        if any(not math.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("Sample weights must be finite and positive")

        self.mean_ = [sum(row[j] for row in matrix) / len(matrix) for j in range(width)]
        self.scale_ = []
        for j in range(width):
            variance = sum((row[j] - self.mean_[j]) ** 2 for row in matrix) / len(matrix)
            scale = math.sqrt(variance)
            self.scale_.append(scale if scale >= 1e-12 else 1.0)
        normalized = [
            [(row[j] - self.mean_[j]) / self.scale_[j] for j in range(width)]
            for row in matrix
        ]
        self.coef_ = [0.0] * width
        self.intercept_ = 0.0
        weight_sum = sum(weights)
        for _ in range(self.iterations):
            gradients = [0.0] * width
            intercept_gradient = 0.0
            for row, label, weight in zip(normalized, target, weights):
                logit = max(-35.0, min(35.0, self.intercept_ + sum(
                    coefficient * value for coefficient, value in zip(self.coef_, row)
                )))
                probability = 1.0 / (1.0 + math.exp(-logit))
                residual = (probability - label) * weight
                for j, value in enumerate(row):
                    gradients[j] += residual * value
                intercept_gradient += residual
            for j in range(width):
                gradients[j] = gradients[j] / weight_sum + self.l2 * self.coef_[j]
                self.coef_[j] -= self.learning_rate * gradients[j]
            self.intercept_ -= self.learning_rate * intercept_gradient / weight_sum
        if not all(
            math.isfinite(value)
            for value in (*self.mean_, *self.scale_, *self.coef_, self.intercept_)
        ):
            raise ValueError("Fitted calibrator contains non-finite parameters")
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        if not self.coef_:
            raise ValueError("Calibrator is not fitted")
        output: list[list[float]] = []
        for raw in rows:
            values = [float(value) for value in raw]
            if len(values) != len(self.coef_) or not all(math.isfinite(value) for value in values):
                raise ValueError("Prediction features are incompatible or non-finite")
            normalized = [
                (value - self.mean_[j]) / self.scale_[j] for j, value in enumerate(values)
            ]
            logit = max(-35.0, min(35.0, self.intercept_ + sum(
                coefficient * value for coefficient, value in zip(self.coef_, normalized)
            )))
            positive = min(1.0 - 1e-12, max(1e-12, 1.0 / (1.0 + math.exp(-logit))))
            output.append([1.0 - positive, positive])
        return output

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": "deterministic_weighted_logistic_v1",
            "mean": list(self.mean_),
            "scale": list(self.scale_),
            "coefficients": list(self.coef_),
            "intercept": self.intercept_,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeterministicLogisticCalibrator":
        if payload.get("model_type") != "deterministic_weighted_logistic_v1":
            raise ValueError("Unsupported calibrator model_type")
        model = cls(
            iterations=int(payload["iterations"]),
            learning_rate=float(payload["learning_rate"]),
            l2=float(payload["l2"]),
        )
        model.mean_ = [float(value) for value in payload["mean"]]
        model.scale_ = [float(value) for value in payload["scale"]]
        model.coef_ = [float(value) for value in payload["coefficients"]]
        model.intercept_ = float(payload["intercept"])
        width = len(model.coef_)
        if width == 0 or len(model.mean_) != width or len(model.scale_) != width:
            raise ValueError("Calibrator parameter dimensions do not match")
        if any(value <= 0.0 for value in model.scale_) or not all(
            math.isfinite(value)
            for value in (*model.mean_, *model.scale_, *model.coef_, model.intercept_)
        ):
            raise ValueError("Calibrator parameters are invalid")
        return model


_CASE_KEYS = {
    "case_id",
    "patient_id",
    "family_id",
    "group_id",
    "split",
    "observations",
    "gold_disease_ids",
    "reference_standard",
    "review_status",
    "synthetic",
    "cohort_id",
    "hpo_release",
}
_OBSERVATION_KEYS = {
    "hpo_id",
    "status",
    "subject",
    "reviewer_status",
    "extraction_confidence",
    "linking_confidence",
    "assertion_certainty",
    "gestational_age_weeks",
}


def _error(
    code: CalibrationErrorCode, message: str, path: str, **details: Any
) -> CalibrationDataError:
    return CalibrationDataError(code, message, path=path, details=details)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _strict_mapping(
    value: Any,
    *,
    path: str,
    allowed: set[str],
    required: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(CalibrationErrorCode.INVALID_FIELD, "Expected an object.", path)
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise _error(
            CalibrationErrorCode.UNKNOWN_FIELD,
            f"Unknown field(s): {', '.join(unknown)}.",
            path,
            fields=unknown,
        )
    missing = sorted(key for key in required if key not in value)
    if missing:
        raise _error(
            CalibrationErrorCode.MISSING_FIELD,
            f"Missing field(s): {', '.join(missing)}.",
            path,
            fields=missing,
        )
    return value


def _string(value: Any, *, path: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise _error(
            CalibrationErrorCode.INVALID_FIELD,
            f"Expected a non-empty string no longer than {maximum} characters.",
            path,
        )
    return value.strip()


def _pseudonymous_id(value: Any, *, path: str) -> str:
    identifier = _string(value, path=path, maximum=64)
    if (
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{2,63}", identifier)
        or re.search(r"\d{9,}", identifier)
        or "@" in identifier
    ):
        raise _error(
            CalibrationErrorCode.DIRECT_IDENTIFIER_RISK,
            "Identifier must be a pseudonymous study code, not an MRN/contact identifier.",
            path,
        )
    return identifier


def _probability(value: Any, *, path: str, nullable: bool = True) -> Optional[float]:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise _error(CalibrationErrorCode.INVALID_FIELD, "Expected a number in [0, 1].", path)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _error(
            CalibrationErrorCode.INVALID_FIELD, "Expected a finite number in [0, 1].", path
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise _error(
            CalibrationErrorCode.INVALID_FIELD, "Expected a finite number in [0, 1].", path
        )
    return number


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise _error(CalibrationErrorCode.INVALID_JSONL, "Calibration JSONL file not found.", "$")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue

        def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise _error(
                        CalibrationErrorCode.INVALID_JSONL,
                        f"Duplicate JSON key {key!r}.",
                        f"line[{line_number}]",
                    )
                result[key] = value
            return result

        try:
            value = json.loads(
                line,
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    _error(
                        CalibrationErrorCode.INVALID_JSONL,
                        f"Non-finite number {constant!r} is forbidden.",
                        f"line[{line_number}]",
                    )
                ),
            )
        except CalibrationDataError:
            raise
        except json.JSONDecodeError as exc:
            raise _error(
                CalibrationErrorCode.INVALID_JSONL,
                f"Invalid JSON on line {line_number}: {exc}",
                f"line[{line_number}]",
            ) from exc
        if not isinstance(value, Mapping):
            raise _error(
                CalibrationErrorCode.INVALID_JSONL,
                "Every non-empty JSONL line must be an object.",
                f"line[{line_number}]",
            )
        rows.append(value)
    if not rows:
        raise _error(CalibrationErrorCode.INVALID_JSONL, "Calibration JSONL is empty.", "$")
    return rows


def _matcher_release(matcher: HPOAgent2Matcher) -> Optional[str]:
    return (matcher.provenance.get("hp_obo") or {}).get("version") or matcher.ontology.version


def _parse_case(
    raw: Mapping[str, Any], *, index: int, matcher: HPOAgent2Matcher, real_mode: bool
) -> CalibrationCase:
    path = f"$[{index}]"
    item = _strict_mapping(raw, path=path, allowed=_CASE_KEYS, required=_CASE_KEYS)
    case_id = _pseudonymous_id(item["case_id"], path=f"{path}.case_id")
    patient_id = _pseudonymous_id(item["patient_id"], path=f"{path}.patient_id")
    family_id = _pseudonymous_id(item["family_id"], path=f"{path}.family_id")
    group_id = _pseudonymous_id(item["group_id"], path=f"{path}.group_id")
    split = item["split"]
    if split not in SPLITS:
        raise _error(
            CalibrationErrorCode.INVALID_FIELD,
            "split must be train, validation or test.",
            f"{path}.split",
        )
    if item["review_status"] != "approved":
        raise _error(
            CalibrationErrorCode.REVIEW_NOT_APPROVED,
            "Patient-level case review_status must be approved.",
            f"{path}.review_status",
        )
    if not isinstance(item["synthetic"], bool):
        raise _error(
            CalibrationErrorCode.INVALID_FIELD, "synthetic must be boolean.", f"{path}.synthetic"
        )
    synthetic = item["synthetic"]
    if real_mode and synthetic:
        raise _error(
            CalibrationErrorCode.DATASET_MODE_MISMATCH,
            "Synthetic cases are forbidden in real mode.",
            f"{path}.synthetic",
        )
    if not real_mode and not synthetic:
        raise _error(
            CalibrationErrorCode.DATASET_MODE_MISMATCH,
            "Non-synthetic cases require explicit real_mode=True.",
            f"{path}.synthetic",
        )
    cohort_id = _string(item["cohort_id"], path=f"{path}.cohort_id", maximum=128)
    hpo_release = _string(item["hpo_release"], path=f"{path}.hpo_release", maximum=128)
    loaded_release = _matcher_release(matcher)
    if loaded_release and hpo_release != loaded_release:
        raise _error(
            CalibrationErrorCode.HPO_RELEASE_MISMATCH,
            "Case HPO release does not match the loaded matcher snapshot.",
            f"{path}.hpo_release",
            case_release=hpo_release,
            matcher_release=loaded_release,
        )

    reference = _strict_mapping(
        item["reference_standard"],
        path=f"{path}.reference_standard",
        allowed={"type", "source", "confirmed_by", "confirmed_at"},
        required={"type", "source", "confirmed_by"},
    )
    reference_type = reference["type"]
    if reference_type not in REFERENCE_STANDARD_TYPES:
        raise _error(
            CalibrationErrorCode.INVALID_FIELD,
            "Unsupported reference_standard.type.",
            f"{path}.reference_standard.type",
        )
    reference_standard = ReferenceStandard(
        type=str(reference_type),
        source=_string(reference["source"], path=f"{path}.reference_standard.source"),
        confirmed_by=_string(
            reference["confirmed_by"], path=f"{path}.reference_standard.confirmed_by"
        ),
        confirmed_at=(
            None
            if reference.get("confirmed_at") is None
            else _string(
                reference["confirmed_at"], path=f"{path}.reference_standard.confirmed_at"
            )
        ),
    )

    raw_observations = item["observations"]
    if not isinstance(raw_observations, list) or not raw_observations:
        raise _error(
            CalibrationErrorCode.INVALID_FIELD,
            "observations must be a non-empty array.",
            f"{path}.observations",
        )
    observations: list[PhenotypeObservation] = []
    seen_status: dict[str, PhenotypeStatus] = {}
    for observation_index, raw_observation in enumerate(raw_observations):
        observation_path = f"{path}.observations[{observation_index}]"
        observation = _strict_mapping(
            raw_observation,
            path=observation_path,
            allowed=_OBSERVATION_KEYS,
            required={"hpo_id", "status", "subject", "reviewer_status"},
        )
        if observation["subject"] != "fetus":
            raise _error(
                CalibrationErrorCode.INVALID_FIELD,
                "Calibration observations must belong to the fetus.",
                f"{observation_path}.subject",
            )
        if observation["reviewer_status"] != "approved":
            raise _error(
                CalibrationErrorCode.REVIEW_NOT_APPROVED,
                "Every observation must be clinician-approved.",
                f"{observation_path}.reviewer_status",
            )
        try:
            parsed = PhenotypeObservation(
                hpo_id=str(observation["hpo_id"]),
                status=PhenotypeStatus.coerce(observation["status"]),
                subject=PhenotypeSubject.FETUS,
                reviewer_status=ReviewerStatus.CLINICIAN_CONFIRMED,
                extraction_confidence=_probability(
                    observation.get("extraction_confidence"),
                    path=f"{observation_path}.extraction_confidence",
                ),
                linking_confidence=_probability(
                    observation.get("linking_confidence"),
                    path=f"{observation_path}.linking_confidence",
                ),
                assertion_certainty=float(
                    _probability(
                        observation.get("assertion_certainty", 1.0),
                        path=f"{observation_path}.assertion_certainty",
                        nullable=False,
                    )
                ),
                gestational_age_weeks=(
                    None
                    if observation.get("gestational_age_weeks") is None
                    else float(observation["gestational_age_weeks"])
                ),
            )
        except CalibrationDataError:
            raise
        except (TypeError, ValueError) as exc:
            raise _error(
                CalibrationErrorCode.INVALID_FIELD, str(exc), observation_path
            ) from exc
        hpo_id = parsed.hpo_id
        if not matcher.ontology.contains(hpo_id) or not matcher.ontology.is_within_phenotypic_abnormality(
            hpo_id
        ):
            raise _error(
                CalibrationErrorCode.UNKNOWN_HPO,
                "HPO term is absent from the loaded phenotypic-abnormality ontology.",
                f"{observation_path}.hpo_id",
                hpo_id=hpo_id,
            )
        if hpo_id in seen_status:
            code = (
                CalibrationErrorCode.DUPLICATE_HPO
                if seen_status[hpo_id] == parsed.status
                else CalibrationErrorCode.CONTRADICTORY_HPO
            )
            raise _error(
                code,
                "Duplicate or contradictory HPO observations are forbidden per patient case.",
                observation_path,
                hpo_id=hpo_id,
                first_status=seen_status[hpo_id].value,
                second_status=parsed.status.value,
            )
        seen_status[hpo_id] = parsed.status
        observations.append(parsed)
    if not any(
        observation.status in {PhenotypeStatus.PRESENT, PhenotypeStatus.SUSPECTED}
        and observation.hpo_id != "HP:0000118"
        for observation in observations
    ):
        raise _error(
            CalibrationErrorCode.NO_POSITIVE_FINDING,
            "At least one informative PRESENT or SUSPECTED fetal HPO is required.",
            f"{path}.observations",
        )

    gold = item["gold_disease_ids"]
    if not isinstance(gold, list) or not gold or any(not isinstance(value, str) for value in gold):
        raise _error(
            CalibrationErrorCode.INVALID_FIELD,
            "gold_disease_ids must be a non-empty array of strings.",
            f"{path}.gold_disease_ids",
        )
    gold_ids = tuple(str(value).strip() for value in gold)
    if any(not value for value in gold_ids) or len(set(gold_ids)) != len(gold_ids):
        raise _error(
            CalibrationErrorCode.INVALID_FIELD,
            "gold_disease_ids must contain unique non-empty IDs.",
            f"{path}.gold_disease_ids",
        )
    known_diseases = {profile.disease_id for profile in matcher.profiles}
    unknown_gold = sorted(set(gold_ids) - known_diseases)
    if unknown_gold:
        raise _error(
            CalibrationErrorCode.UNKNOWN_GOLD_DISEASE,
            "Gold disease is absent from the loaded matcher disease universe.",
            f"{path}.gold_disease_ids",
            disease_ids=unknown_gold,
        )
    return CalibrationCase(
        case_id=case_id,
        patient_id=patient_id,
        family_id=family_id,
        group_id=group_id,
        split=str(split),
        observations=tuple(observations),
        gold_disease_ids=gold_ids,
        reference_standard=reference_standard,
        review_status="approved",
        synthetic=synthetic,
        cohort_id=cohort_id,
        hpo_release=hpo_release,
    )


def validate_calibration_cases(
    source_path: str | Path,
    matcher: HPOAgent2Matcher,
    *,
    real_mode: bool = False,
    expected_source_sha256: Optional[str] = None,
) -> ValidatedCalibrationDataset:
    """Validate a complete three-way patient-level split before feature creation."""

    path = Path(source_path).resolve()
    source_sha256 = _sha256_file(path)
    if expected_source_sha256 is not None:
        expected = str(expected_source_sha256).strip().casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or not hmac.compare_digest(
            source_sha256, expected
        ):
            raise _error(
                CalibrationErrorCode.SOURCE_INTEGRITY_MISMATCH,
                "Calibration source does not match expected SHA-256.",
                "$",
                expected_sha256=expected,
                actual_sha256=source_sha256,
            )
    raw_rows = _read_jsonl(path)
    cases = tuple(
        _parse_case(raw, index=index, matcher=matcher, real_mode=real_mode)
        for index, raw in enumerate(raw_rows)
    )
    split_names = {case.split for case in cases}
    missing_splits = sorted(set(SPLITS) - split_names)
    if missing_splits:
        raise _error(
            CalibrationErrorCode.SPLIT_MISSING,
            "Train, validation and test must all be present.",
            "$",
            missing_splits=missing_splits,
        )

    seen_cases: set[str] = set()
    patient_split: dict[str, str] = {}
    family_split: dict[str, str] = {}
    group_split: dict[str, str] = {}
    for case in cases:
        if case.case_id in seen_cases:
            raise _error(
                CalibrationErrorCode.DUPLICATE_CASE,
                "case_id must be globally unique.",
                "$",
                case_id=case.case_id,
            )
        seen_cases.add(case.case_id)
        if case.patient_id in patient_split:
            code = (
                CalibrationErrorCode.SPLIT_LEAKAGE
                if patient_split[case.patient_id] != case.split
                else CalibrationErrorCode.DUPLICATE_PATIENT
            )
            raise _error(
                code,
                "A patient may appear in exactly one patient-level case and one split.",
                "$",
                patient_id=case.patient_id,
                first_split=patient_split[case.patient_id],
                second_split=case.split,
            )
        patient_split[case.patient_id] = case.split
        for label, identifier, registry in (
            ("family_id", case.family_id, family_split),
            ("group_id", case.group_id, group_split),
        ):
            prior_split = registry.get(identifier)
            if prior_split is not None and prior_split != case.split:
                raise _error(
                    CalibrationErrorCode.SPLIT_LEAKAGE,
                    f"{label} crosses train/validation/test boundaries.",
                    "$",
                    identifier=identifier,
                    first_split=prior_split,
                    second_split=case.split,
                )
            registry[identifier] = case.split
    return ValidatedCalibrationDataset(
        cases=cases,
        source_path=path,
        source_sha256=source_sha256,
        real_mode=real_mode,
        matcher_hpo_release=_matcher_release(matcher),
    )


def build_candidate_feature_rows(
    dataset: ValidatedCalibrationDataset,
    matcher: HPOAgent2Matcher,
    *,
    train_negative_count: int = 100,
    random_state: int = 17,
) -> CandidateFeatureDataset:
    """Create deterministic rows; validation/test always use the full universe.

    Train negatives are selected by a label-independent SHA-256 ordering and
    receive inverse sampling weights.  Gold diseases are always retained.
    Validation and locked test metrics are computed on every matcher disease,
    not on a top-k selected using the same score being evaluated.
    """

    if train_negative_count < 1:
        raise ValueError("train_negative_count must be at least 1")
    universe = tuple(sorted(profile.disease_id for profile in matcher.profiles))
    universe_set = set(universe)
    rows: list[CandidateFeatureRow] = []
    for case in sorted(dataset.cases, key=lambda item: (SPLITS.index(item.split), item.case_id)):
        gold = set(case.gold_disease_ids)
        if not gold <= universe_set:
            raise ValueError("Validated gold disease set no longer matches matcher universe")
        negatives = [disease_id for disease_id in universe if disease_id not in gold]
        if not negatives:
            raise _error(
                CalibrationErrorCode.CLASS_MISSING,
                "Disease universe provides no negative candidate for a case.",
                "$",
                case_id=case.case_id,
            )
        if case.split == "train":
            ordered_negatives = sorted(
                negatives,
                key=lambda disease_id: hashlib.sha256(
                    f"{random_state}\0{case.case_id}\0{disease_id}".encode("utf-8")
                ).hexdigest(),
            )
            selected_negatives = ordered_negatives[: min(train_negative_count, len(negatives))]
            selected = tuple(sorted(gold)) + tuple(selected_negatives)
            negative_weight = len(negatives) / len(selected_negatives)
        else:
            selected = universe
            negative_weight = 1.0
        ranked = matcher.rank(case.observations, disease_ids=selected, top_k=None)
        by_disease = {candidate.disease_id: candidate for candidate in ranked}
        missing_gold = sorted(gold - set(by_disease))
        if missing_gold:
            raise _error(
                CalibrationErrorCode.UNKNOWN_GOLD_DISEASE,
                "Gold candidate was lost during feature generation.",
                "$",
                case_id=case.case_id,
                disease_ids=missing_gold,
            )
        for disease_id in selected:
            candidate = by_disease.get(disease_id)
            if candidate is None:
                raise ValueError(f"Matcher did not return requested disease {disease_id!r}")
            features = candidate.feature_vector()
            if not all(math.isfinite(value) for value in features):
                raise ValueError("Candidate feature row contains NaN or infinity")
            label = int(disease_id in gold)
            rows.append(
                CandidateFeatureRow(
                    case_id=case.case_id,
                    patient_id=case.patient_id,
                    family_id=case.family_id,
                    group_id=case.group_id,
                    split=case.split,
                    disease_id=disease_id,
                    label=label,
                    features=features,
                    sample_weight=1.0 if label else negative_weight,
                    selection_reason=(
                        "gold_always_included"
                        if label
                        else (
                            "deterministic_hash_negative_with_inverse_sampling_weight"
                            if case.split == "train"
                            else "full_evaluation_disease_universe"
                        )
                    ),
                )
            )
    result = CandidateFeatureDataset(
        rows=tuple(rows),
        feature_names=tuple(matcher.FEATURE_NAMES),
        feature_version=FEATURE_VERSION,
        random_state=int(random_state),
        train_negative_count=int(train_negative_count),
        negative_sampling_description=(
            "Train: every gold candidate plus up to "
            f"{train_negative_count} negatives selected by deterministic label-independent SHA-256 "
            "ordering; negatives carry inverse sampling weights. Validation/test: full matcher "
            "disease universe. Metrics are conditional on this candidate construction and are not "
            "population disease prevalence."
        ),
    )
    for split, split_rows in result.by_split.items():
        labels = {row.label for row in split_rows}
        if labels != {0, 1}:
            raise _error(
                CalibrationErrorCode.CLASS_MISSING,
                f"{split} candidate rows must contain positive and negative labels.",
                "$",
                split=split,
            )
    return result


def _predictions(
    model: DeterministicLogisticCalibrator, rows: Sequence[CandidateFeatureRow]
) -> list[float]:
    probabilities = [pair[1] for pair in model.predict_proba([row.features for row in rows])]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("Calibrator returned NaN, infinity or an out-of-range probability")
    return probabilities


def _auc(labels: Sequence[int], probabilities: Sequence[float]) -> Optional[float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(probabilities, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def calibration_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float,
    ece_bins: int = 10,
) -> dict[str, Optional[float]]:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Metrics require aligned non-empty labels and probabilities")
    clipped = [min(1.0 - 1e-12, max(1e-12, float(value))) for value in probabilities]
    target = [int(value) for value in labels]
    if any(value not in {0, 1} for value in target):
        raise ValueError("Metrics labels must be binary")
    count = len(target)
    brier = sum((probability - label) ** 2 for probability, label in zip(clipped, target)) / count
    log_loss = -sum(
        label * math.log(probability) + (1 - label) * math.log(1.0 - probability)
        for probability, label in zip(clipped, target)
    ) / count
    ece = 0.0
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        members = [
            index
            for index, probability in enumerate(clipped)
            if lower <= probability < upper or (bin_index == ece_bins - 1 and probability == 1.0)
        ]
        if members:
            confidence = sum(clipped[index] for index in members) / len(members)
            frequency = sum(target[index] for index in members) / len(members)
            ece += len(members) / count * abs(confidence - frequency)
    predicted = [int(probability >= threshold) for probability in clipped]
    tp = sum(p == 1 and y == 1 for p, y in zip(predicted, target))
    tn = sum(p == 0 and y == 0 for p, y in zip(predicted, target))
    fp = sum(p == 1 and y == 0 for p, y in zip(predicted, target))
    fn = sum(p == 0 and y == 1 for p, y in zip(predicted, target))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    metrics: dict[str, Optional[float]] = {
        "row_count": float(count),
        "positive_count": float(sum(target)),
        "brier": float(brier),
        "log_loss": float(log_loss),
        "ece": float(ece),
        "auc": _auc(target, clipped),
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
    if not all(value is None or math.isfinite(value) for value in metrics.values()):
        raise ValueError("Calibration metrics contain NaN or infinity")
    return metrics


def select_threshold_on_validation(
    labels: Sequence[int], probabilities: Sequence[float]
) -> float:
    """Select threshold on validation only; test labels are not accepted here."""

    candidates = sorted({0.5, *(float(value) for value in probabilities)})
    best: tuple[float, float, float] | None = None
    best_threshold = 0.5
    for threshold in candidates:
        metrics = calibration_metrics(labels, probabilities, threshold=threshold)
        sensitivity = metrics["sensitivity"]
        specificity = metrics["specificity"]
        balanced = (
            (float(sensitivity) + float(specificity)) / 2.0
            if sensitivity is not None and specificity is not None
            else -1.0
        )
        key = (balanced, -abs(threshold - 0.5), -threshold)
        if best is None or key > best:
            best = key
            best_threshold = threshold
    return float(best_threshold)


def _validate_attestation(
    value: Optional[Mapping[str, Any]], *, dataset_sha256: str
) -> tuple[Optional[str], Optional[str]]:
    if value is None:
        return None, None
    required = {
        "approval_id",
        "approved_by",
        "approved_at",
        "dataset_sha256",
        "clinical_ready_approved",
        "independent_test_locked",
    }
    item = _strict_mapping(value, path="$.attestation", allowed=required, required=required)
    if item["clinical_ready_approved"] is not True or item["independent_test_locked"] is not True:
        raise _error(
            CalibrationErrorCode.ATTESTATION_INVALID,
            "Attestation must explicitly approve clinical readiness and locked independent test.",
            "$.attestation",
        )
    if str(item["dataset_sha256"]).casefold() != dataset_sha256.casefold():
        raise _error(
            CalibrationErrorCode.ATTESTATION_INVALID,
            "Attestation is not bound to this dataset SHA-256.",
            "$.attestation.dataset_sha256",
        )
    approval_id = _pseudonymous_id(item["approval_id"], path="$.attestation.approval_id")
    _pseudonymous_id(item["approved_by"], path="$.attestation.approved_by")
    _string(item["approved_at"], path="$.attestation.approved_at")
    return _sha256_json(dict(item)), approval_id


def train_agent2_calibrator(
    dataset: ValidatedCalibrationDataset,
    matcher: HPOAgent2Matcher,
    *,
    train_negative_count: int = 100,
    random_state: int = 17,
    approval_attestation: Optional[Mapping[str, Any]] = None,
) -> CalibrationTrainingResult:
    """Fit train-only, choose threshold on validation, evaluate locked test once."""

    if dataset.real_mode:
        if approval_attestation is None:
            attestation_sha256 = approval_id = None
        else:
            attestation_sha256, approval_id = _validate_attestation(
                approval_attestation, dataset_sha256=dataset.source_sha256
            )
    else:
        if approval_attestation is not None:
            raise _error(
                CalibrationErrorCode.ATTESTATION_INVALID,
                "Synthetic data can never be promoted by an approval attestation.",
                "$.attestation",
            )
        attestation_sha256 = approval_id = None

    features = build_candidate_feature_rows(
        dataset,
        matcher,
        train_negative_count=train_negative_count,
        random_state=random_state,
    )
    split_rows = features.by_split
    train_rows = split_rows["train"]
    validation_rows = split_rows["validation"]
    test_rows = split_rows["test"]
    model = DeterministicLogisticCalibrator().fit(
        [row.features for row in train_rows],
        [row.label for row in train_rows],
        [row.sample_weight for row in train_rows],
    )
    validation_probabilities = _predictions(model, validation_rows)
    threshold = select_threshold_on_validation(
        [row.label for row in validation_rows], validation_probabilities
    )
    validation_metrics = calibration_metrics(
        [row.label for row in validation_rows],
        validation_probabilities,
        threshold=threshold,
    )
    # Locked test is deliberately touched only after model and threshold are final.
    test_probabilities = _predictions(model, test_rows)
    test_metrics = calibration_metrics(
        [row.label for row in test_rows], test_probabilities, threshold=threshold
    )
    clinical_approved = bool(dataset.real_mode and attestation_sha256 and approval_id)
    cohorts = dataset.by_split
    metadata_metrics = {
        **{
            f"validation_{key}": float(value)
            for key, value in validation_metrics.items()
            if value is not None
        },
        **{
            f"test_{key}": float(value)
            for key, value in test_metrics.items()
            if value is not None
        },
    }
    metadata = CalibratorMetadata(
        method="deterministic_weighted_logistic_v1",
        feature_names=tuple(matcher.FEATURE_NAMES),
        feature_version=FEATURE_VERSION,
        hpo_release=dataset.matcher_hpo_release,
        disease_universe=f"{len(matcher.profiles)} loaded HPOA disease profiles",
        training_cohort=",".join(sorted({case.cohort_id for case in cohorts["train"]})),
        validation_cohort=",".join(
            sorted({case.cohort_id for case in cohorts["validation"]})
        ),
        test_cohort=",".join(sorted({case.cohort_id for case in cohorts["test"]})),
        split_unit="family",
        synthetic_only=not dataset.real_mode,
        clinician_reviewed=clinical_approved,
        negative_sampling=features.negative_sampling_description,
        metrics=metadata_metrics,
        fitted_at=datetime.now(timezone.utc).isoformat(),
    )
    if not dataset.real_mode and metadata.clinical_ready:
        raise AssertionError("Synthetic calibrator must never be clinical_ready")
    return CalibrationTrainingResult(
        model=model,
        metadata=metadata,
        threshold=threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        feature_dataset=features,
        approval_attestation_sha256=attestation_sha256,
        approval_id=approval_id,
        test_evaluation_count=1,
    )


def _metadata_from_dict(payload: Mapping[str, Any]) -> CalibratorMetadata:
    return CalibratorMetadata(
        method=str(payload["method"]),
        feature_names=tuple(payload["feature_names"]),
        feature_version=str(payload["feature_version"]),
        hpo_release=payload.get("hpo_release"),
        disease_universe=str(payload["disease_universe"]),
        training_cohort=str(payload["training_cohort"]),
        validation_cohort=payload.get("validation_cohort"),
        test_cohort=payload.get("test_cohort"),
        split_unit=str(payload["split_unit"]),
        synthetic_only=bool(payload["synthetic_only"]),
        clinician_reviewed=bool(payload["clinician_reviewed"]),
        negative_sampling=payload.get("negative_sampling"),
        metrics=dict(payload.get("metrics") or {}),
        fitted_at=payload.get("fitted_at"),
        schema_version=str(payload.get("schema_version", "agent2-calibrator/1")),
    )


def save_calibrator_artifact(
    result: CalibrationTrainingResult,
    dataset: ValidatedCalibrationDataset,
    matcher: HPOAgent2Matcher,
    output_dir: str | Path,
    *,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> Path:
    """Write a new non-overwriting JSON artifact directory and manifest."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_EXISTS,
            "Output directory already exists; refusing to overwrite.",
            path=str(output),
        )
    schema = Path(schema_path).resolve()
    if not schema.is_file():
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH, "Calibration JSON schema is missing."
        )
    model_payload = result.model.as_dict()
    metadata_payload = result.metadata.as_dict()
    evaluation_payload = {
        **result.summary(),
        "scope": (
            "Candidate-level correctness under the documented disease universe and sampling. "
            "Not population prevalence and not a diagnosis."
        ),
    }
    model_bytes = _canonical_json_bytes(model_payload)
    metadata_bytes = _canonical_json_bytes(metadata_payload)
    evaluation_bytes = _canonical_json_bytes(evaluation_payload)
    provenance = matcher.provenance
    hpoa = dict(provenance.get("phenotype_hpoa") or {})
    obo = dict(provenance.get("hp_obo") or {})
    manifest = {
        "artifact_version": CALIBRATOR_ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "filename": dataset.source_path.name,
            "sha256": dataset.source_sha256,
            "real_mode": dataset.real_mode,
        },
        "schema": {"filename": schema.name, "sha256": _sha256_file(schema)},
        "matcher": {
            "hpoa_filename": hpoa.get("filename"),
            "hpoa_sha256": hpoa.get("sha256"),
            "obo_filename": obo.get("filename"),
            "obo_sha256": obo.get("sha256"),
            "hpo_release": dataset.matcher_hpo_release,
            "disease_count": len(matcher.profiles),
        },
        "splits": dict(dataset.split_sha256),
        "features": {
            "version": FEATURE_VERSION,
            "names": list(matcher.FEATURE_NAMES),
            "contract_sha256": _sha256_json(
                {"version": FEATURE_VERSION, "names": list(matcher.FEATURE_NAMES)}
            ),
            "candidate_rows_sha256": result.feature_dataset.sha256,
            "random_state": result.feature_dataset.random_state,
            "train_negative_count": result.feature_dataset.train_negative_count,
            "negative_sampling": result.feature_dataset.negative_sampling_description,
        },
        "approval": {
            "attestation_sha256": result.approval_attestation_sha256,
            "approval_id": result.approval_id,
            "clinical_ready": result.metadata.clinical_ready,
        },
        "locked_test": {
            "evaluation_count": result.test_evaluation_count,
            "threshold_source": "validation",
        },
        "files": {
            "model.json": _sha256_bytes(model_bytes),
            "metadata.json": _sha256_bytes(metadata_bytes),
            "evaluation.json": _sha256_bytes(evaluation_bytes),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "model.json").write_bytes(model_bytes)
    (output / "metadata.json").write_bytes(metadata_bytes)
    (output / "evaluation.json").write_bytes(evaluation_bytes)
    (output / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
    return output


def load_calibrator_artifact(
    artifact_dir: str | Path,
    matcher: HPOAgent2Matcher,
    source_path: str | Path,
    *,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    attach: bool = True,
    expected_manifest_sha256: Optional[str] = None,
) -> tuple[DeterministicLogisticCalibrator, CalibratorMetadata, Mapping[str, Any]]:
    """Verify all recorded digests/versions before optionally attaching a model."""

    root = Path(artifact_dir).resolve()
    manifest_path = root / "manifest.json"
    if expected_manifest_sha256 is not None:
        expected = str(expected_manifest_sha256).strip().casefold()
        actual = _sha256_file(manifest_path)
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or not hmac.compare_digest(
            expected, actual
        ):
            raise CalibrationArtifactError(
                CalibrationErrorCode.ARTIFACT_MISMATCH,
                "Artifact manifest does not match the trusted SHA-256.",
                details={"expected_sha256": expected, "actual_sha256": actual},
            )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH, f"Cannot read artifact manifest: {exc}"
        ) from exc
    if manifest.get("artifact_version") != CALIBRATOR_ARTIFACT_VERSION:
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH, "Unsupported artifact version."
        )
    for filename in ("model.json", "metadata.json", "evaluation.json"):
        path = root / filename
        if not path.is_file() or _sha256_file(path) != manifest.get("files", {}).get(filename):
            raise CalibrationArtifactError(
                CalibrationErrorCode.ARTIFACT_MISMATCH,
                f"Artifact file digest mismatch: {filename}.",
            )
    source = Path(source_path).resolve()
    if _sha256_file(source) != manifest.get("source", {}).get("sha256"):
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH, "Calibration source digest mismatch."
        )
    schema = Path(schema_path).resolve()
    if _sha256_file(schema) != manifest.get("schema", {}).get("sha256"):
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH, "Calibration schema digest mismatch."
        )
    if (
        manifest.get("features", {}).get("version") != FEATURE_VERSION
        or tuple(manifest.get("features", {}).get("names") or ()) != tuple(matcher.FEATURE_NAMES)
        or manifest.get("features", {}).get("contract_sha256")
        != _sha256_json({"version": FEATURE_VERSION, "names": list(matcher.FEATURE_NAMES)})
    ):
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH, "Feature version/name mismatch."
        )
    current_hpoa = dict(matcher.provenance.get("phenotype_hpoa") or {})
    current_obo = dict(matcher.provenance.get("hp_obo") or {})
    recorded_matcher = manifest.get("matcher", {})
    for label, current, recorded in (
        ("HPOA", current_hpoa.get("sha256"), recorded_matcher.get("hpoa_sha256")),
        ("OBO", current_obo.get("sha256"), recorded_matcher.get("obo_sha256")),
    ):
        if current != recorded:
            raise CalibrationArtifactError(
                CalibrationErrorCode.ARTIFACT_MISMATCH, f"{label} snapshot digest mismatch."
            )
    real_mode = manifest.get("source", {}).get("real_mode")
    if not isinstance(real_mode, bool):
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH, "Artifact source mode is invalid."
        )
    dataset = validate_calibration_cases(source, matcher, real_mode=real_mode)
    if dict(dataset.split_sha256) != dict(manifest.get("splits") or {}):
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH, "Patient split digest mismatch."
        )
    feature_dataset = build_candidate_feature_rows(
        dataset,
        matcher,
        train_negative_count=int(manifest["features"]["train_negative_count"]),
        random_state=int(manifest["features"]["random_state"]),
    )
    if feature_dataset.sha256 != manifest["features"].get("candidate_rows_sha256"):
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH, "Candidate feature rows digest mismatch."
        )
    try:
        model = DeterministicLogisticCalibrator.from_dict(
            json.loads((root / "model.json").read_text(encoding="utf-8"))
        )
        metadata = _metadata_from_dict(
            json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH, f"Invalid model/metadata payload: {exc}"
        ) from exc
    if metadata.synthetic_only == real_mode or metadata.clinical_ready != bool(
        manifest.get("approval", {}).get("clinical_ready")
    ):
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH,
            "Synthetic/real or clinical-ready metadata contradicts the manifest.",
        )
    if metadata.clinical_ready and expected_manifest_sha256 is None:
        raise CalibrationArtifactError(
            CalibrationErrorCode.ARTIFACT_MISMATCH,
            "Clinical-ready artifact loading requires a trusted expected_manifest_sha256.",
        )
    if attach:
        matcher.set_calibrator(model, metadata)
    return model, metadata, manifest


def artifact_manifest_sha256(artifact_dir: str | Path) -> str:
    """Return the trust-anchor digest that must be stored outside the artifact."""

    return _sha256_file(Path(artifact_dir).resolve() / "manifest.json")


__all__ = [
    "CALIBRATION_CASE_SCHEMA_VERSION",
    "CALIBRATOR_ARTIFACT_VERSION",
    "CalibrationArtifactError",
    "CalibrationCase",
    "CalibrationDataError",
    "CalibrationErrorCode",
    "CalibrationTrainingResult",
    "CandidateFeatureDataset",
    "CandidateFeatureRow",
    "DeterministicLogisticCalibrator",
    "ReferenceStandard",
    "ValidatedCalibrationDataset",
    "build_candidate_feature_rows",
    "artifact_manifest_sha256",
    "calibration_metrics",
    "load_calibrator_artifact",
    "save_calibrator_artifact",
    "select_threshold_on_validation",
    "train_agent2_calibrator",
    "validate_calibration_cases",
]
