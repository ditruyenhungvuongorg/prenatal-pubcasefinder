"""Dependency-light, patient-level evaluation utilities for the HPO agents.

Ranking metrics are macro-averaged over unique patients.  This prevents cases
with many phenotype phrases or many returned candidates from receiving more
weight than other patients.  Agent 1 mention metrics are micro counts, but every
span/code key is namespaced by patient before matching.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping, Sequence


_RANKING_ID_FIELDS = (
    "disease_id",
    "gene_symbol",
    "variant_id",
    "hpo_id",
    "code",
    "id",
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        converted = as_dict()
        if isinstance(converted, Mapping):
            return converted
    raise TypeError(f"Expected a mapping or dataclass, got {type(value).__name__}")


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        if "items" in value:
            return _as_sequence(value["items"])
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _item_identifier(item: Any, id_key: str | None = None) -> str:
    if isinstance(item, (str, int)):
        return str(item)
    record = _as_mapping(item)
    fields = (id_key,) if id_key else _RANKING_ID_FIELDS
    for field in fields:
        if field and record.get(field) not in (None, ""):
            return str(record[field])
    raise ValueError(f"Could not find a ranking identifier in fields {fields}: {record}")


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _validate_k_values(values: Sequence[int]) -> tuple[int, ...]:
    try:
        normalized = tuple(sorted({int(value) for value in values}))
    except (TypeError, ValueError) as exc:
        raise ValueError("k_values must contain positive integers") from exc
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("k_values must contain at least one positive integer")
    return normalized


def patient_level_ranking_metrics(
    cases: Sequence[Mapping[str, Any] | Any],
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
    patient_id_key: str = "patient_id",
    truth_key: str = "truth_ids",
    predictions_key: str = "predictions",
    prediction_id_key: str | None = None,
    truth_id_key: str | None = None,
) -> dict[str, Any]:
    """Calculate patient-level Top-k accuracy and mean reciprocal rank.

    A case is a hit when *any* accepted ground-truth identifier occurs in the
    first ``k`` unique predictions.  Empty predictions count as misses.  Empty
    ground truth and duplicate patient IDs raise errors rather than silently
    changing the denominator.
    """

    ks = _validate_k_values(k_values)
    if not cases:
        raise ValueError("At least one patient case is required")

    seen_patients: set[str] = set()
    hit_counts = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    patients_with_hit = 0
    patients_without_predictions = 0
    per_patient: list[dict[str, Any]] = []

    for case_index, raw_case in enumerate(cases):
        case = _as_mapping(raw_case)
        raw_patient_id = case.get(patient_id_key)
        patient_id = "" if raw_patient_id is None else str(raw_patient_id).strip()
        if not patient_id:
            raise ValueError(f"Case {case_index} is missing {patient_id_key!r}")
        if patient_id in seen_patients:
            raise ValueError(f"Duplicate patient ID would bias metrics: {patient_id}")
        seen_patients.add(patient_id)

        truth = {
            _item_identifier(item, truth_id_key)
            for item in _as_sequence(case.get(truth_key))
        }
        if not truth:
            raise ValueError(f"Patient {patient_id} has no accepted ground-truth identifier")
        predictions = _deduplicate(
            _item_identifier(item, prediction_id_key)
            for item in _as_sequence(case.get(predictions_key))
        )
        if not predictions:
            patients_without_predictions += 1

        first_relevant_rank = next(
            (rank for rank, identifier in enumerate(predictions, 1) if identifier in truth),
            None,
        )
        if first_relevant_rank is not None:
            patients_with_hit += 1
            reciprocal_rank_sum += 1.0 / first_relevant_rank
        for k in ks:
            if first_relevant_rank is not None and first_relevant_rank <= k:
                hit_counts[k] += 1
        per_patient.append(
            {
                "patient_id": patient_id,
                "accepted_truth_ids": sorted(truth),
                "prediction_count": len(predictions),
                "first_relevant_rank": first_relevant_rank,
                "reciprocal_rank": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
            }
        )

    patient_count = len(cases)
    return {
        "unit": "patient",
        "patient_count": patient_count,
        "top_k_accuracy": {str(k): hit_counts[k] / patient_count for k in ks},
        "mean_reciprocal_rank": reciprocal_rank_sum / patient_count,
        "patients_with_relevant_prediction": patients_with_hit,
        "patients_without_relevant_prediction": patient_count - patients_with_hit,
        "patients_without_predictions": patients_without_predictions,
        "k_values": list(ks),
        "per_patient": per_patient,
    }


def binary_calibration_metrics(
    scores: Sequence[float],
    labels: Sequence[int | bool],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return Brier score, ECE and reliability-bin data for confidence scores.

    The values are treated as empirical confidence scores, not as disease
    posterior probabilities.  Callers are responsible for evaluating on a
    held-out patient split.
    """

    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and have equal length")
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    normalized_scores: list[float] = []
    normalized_labels: list[int] = []
    for index, (raw_score, raw_label) in enumerate(zip(scores, labels)):
        score = float(raw_score)
        label = int(raw_label)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Confidence score at index {index} must be finite and within [0, 1]")
        if label not in {0, 1}:
            raise ValueError(f"Calibration label at index {index} must be binary")
        normalized_scores.append(score)
        normalized_labels.append(label)

    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for index, score in enumerate(normalized_scores):
        bin_index = min(n_bins - 1, int(score * n_bins))
        bins[bin_index].append(index)

    reliability: list[dict[str, Any]] = []
    expected_calibration_error = 0.0
    sample_count = len(normalized_scores)
    for bin_index, members in enumerate(bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if members:
            mean_confidence = sum(normalized_scores[index] for index in members) / len(members)
            empirical_rate = sum(normalized_labels[index] for index in members) / len(members)
            gap = abs(mean_confidence - empirical_rate)
            expected_calibration_error += len(members) / sample_count * gap
        else:
            mean_confidence = None
            empirical_rate = None
            gap = None
        reliability.append(
            {
                "lower_bound": lower,
                "upper_bound": upper,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "empirical_positive_rate": empirical_rate,
                "calibration_gap": gap,
            }
        )

    brier_score = sum(
        (score - label) ** 2
        for score, label in zip(normalized_scores, normalized_labels)
    ) / sample_count
    return {
        "score_semantics": "empirical_confidence_not_disease_posterior",
        "sample_count": sample_count,
        "positive_count": sum(normalized_labels),
        "brier_score": brier_score,
        "expected_calibration_error": expected_calibration_error,
        "n_bins": n_bins,
        "reliability_bins": reliability,
    }


def evaluate_agent2_rankings(
    cases: Sequence[Mapping[str, Any] | Any],
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Evaluate Agent 2 disease candidates at patient level.

    Each case supplies ``patient_id``, ``truth_disease_ids`` and
    ``agent2_results`` (candidate objects or dictionaries). Calibration metrics
    are computed only from explicitly calibrated disease probabilities. Ranking
    confidence and compatibility scores are never reinterpreted as probability.
    """

    normalized = []
    calibration_scores: list[float] = []
    calibration_labels: list[int] = []
    total_candidate_count = 0
    legacy_probability_count = 0
    ranking_confidence_count = 0
    uncalibrated_probability_count = 0
    for raw_case in cases:
        case = _as_mapping(raw_case)
        truth_values = case.get("truth_disease_ids", case.get("truth_ids"))
        truth_ids = {_item_identifier(item) for item in _as_sequence(truth_values)}
        candidate_values = case.get("agent2_results", case.get("predictions"))
        seen_candidate_ids: set[str] = set()
        for candidate in _as_sequence(candidate_values):
            candidate_record = _as_mapping(candidate)
            disease_id = _item_identifier(candidate_record, "disease_id")
            if disease_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(disease_id)
            total_candidate_count += 1
            if candidate_record.get("confidence_score") is not None:
                ranking_confidence_count += 1
            if candidate_record.get("probability") is not None:
                legacy_probability_count += 1
            probability = candidate_record.get("estimated_probability")
            if probability is None:
                continue
            probability_kind = str(candidate_record.get("probability_kind", "")).upper()
            calibrated = candidate_record.get("calibrated") is True
            if not calibrated or probability_kind != "CALIBRATED_PROBABILITY":
                uncalibrated_probability_count += 1
                continue
            calibration_scores.append(float(probability))
            calibration_labels.append(int(disease_id in truth_ids))
        normalized.append(
            {
                "patient_id": case.get("patient_id", case.get("case_id")),
                "truth_ids": truth_values,
                "predictions": candidate_values,
            }
        )
    report = patient_level_ranking_metrics(
        normalized,
        k_values=k_values,
        prediction_id_key="disease_id",
    )
    report["agent"] = "agent2"
    report["target"] = "disease"
    if calibration_scores:
        report["calibration"] = {
            "status": "available",
            "scored_candidate_count": len(calibration_scores),
            "total_candidate_count": total_candidate_count,
            "legacy_probability_field_count": legacy_probability_count,
            "ranking_confidence_field_count": ranking_confidence_count,
            "uncalibrated_probability_count": uncalibrated_probability_count,
            "probability_field": "estimated_probability",
            "required_probability_kind": "CALIBRATED_PROBABILITY",
            **binary_calibration_metrics(calibration_scores, calibration_labels),
        }
    else:
        report["calibration"] = {
            "status": "not_available",
            "reason": (
                "No candidate had estimated_probability with calibrated=true and "
                "probability_kind=CALIBRATED_PROBABILITY."
            ),
            "scored_candidate_count": 0,
            "total_candidate_count": total_candidate_count,
            "legacy_probability_field_count": legacy_probability_count,
            "ranking_confidence_field_count": ranking_confidence_count,
            "uncalibrated_probability_count": uncalibrated_probability_count,
        }
    return report


def evaluate_agent3_rankings(
    cases: Sequence[Mapping[str, Any] | Any],
    *,
    target: str = "disease",
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Evaluate one Agent 3 ranking stage at patient level.

    ``target`` is ``disease``, ``gene`` or ``variant``.  Each case contains an
    ``agent3_output`` returned by :class:`Agent3Ranker` and the corresponding
    ``truth_*_ids`` collection.
    """

    stage_config = {
        "disease": ("disease_ranking", "truth_disease_ids", "disease_id"),
        "gene": ("gene_ranking", "truth_gene_ids", "gene_symbol"),
        "variant": ("variant_ranking", "truth_variant_ids", "variant_id"),
    }
    if target not in stage_config:
        raise ValueError("target must be disease, gene, or variant")
    stage_key, truth_key, identifier_key = stage_config[target]
    normalized = []
    for raw_case in cases:
        case = _as_mapping(raw_case)
        output = _as_mapping(case.get("agent3_output", case.get("output", {})))
        stage = output.get(stage_key, {})
        normalized.append(
            {
                "patient_id": case.get("patient_id", case.get("case_id", output.get("case_id"))),
                "truth_ids": case.get(truth_key, case.get("truth_ids")),
                "predictions": stage,
            }
        )
    report = patient_level_ranking_metrics(
        normalized,
        k_values=k_values,
        prediction_id_key=identifier_key,
    )
    report["agent"] = "agent3"
    report["target"] = target
    return report


def _mention_span(mention: Mapping[str, Any]) -> tuple[int, int]:
    start = mention.get("span_start", mention.get("start"))
    end = mention.get("span_end", mention.get("end"))
    if start is None or end is None:
        raise ValueError(f"Mention requires span_start/span_end or start/end: {mention}")
    try:
        start_int, end_int = int(start), int(end)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Mention span must be integer-valued: {mention}") from exc
    if start_int < 0 or end_int < start_int:
        raise ValueError(f"Invalid mention span [{start_int}, {end_int})")
    return start_int, end_int


def _mention_code(mention: Mapping[str, Any]) -> str | None:
    value = mention.get("hpo_id", mention.get("code"))
    return None if value in (None, "") else str(value)


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _counter_metric(gold: Counter[Any], predicted: Counter[Any]) -> dict[str, Any]:
    true_positive = sum((gold & predicted).values())
    false_positive = sum((predicted - gold).values())
    false_negative = sum((gold - predicted).values())
    return _prf(true_positive, false_positive, false_negative)


def _maximum_overlap_matches(
    gold: Sequence[tuple[int, int]], predicted: Sequence[tuple[int, int]]
) -> int:
    """Maximum bipartite matches where two half-open spans overlap."""

    edges = {
        pred_index: [
            gold_index
            for gold_index, (gold_start, gold_end) in enumerate(gold)
            if max(gold_start, pred_start) < min(gold_end, pred_end)
        ]
        for pred_index, (pred_start, pred_end) in enumerate(predicted)
    }
    matched_gold: dict[int, int] = {}

    def augment(pred_index: int, visited: set[int]) -> bool:
        for gold_index in edges[pred_index]:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            if gold_index not in matched_gold or augment(matched_gold[gold_index], visited):
                matched_gold[gold_index] = pred_index
                return True
        return False

    return sum(augment(pred_index, set()) for pred_index in range(len(predicted)))


def evaluate_agent1_mentions(
    cases: Sequence[Mapping[str, Any] | Any],
    *,
    patient_id_key: str = "patient_id",
    gold_key: str = "gold_mentions",
    predictions_key: str = "predicted_mentions",
) -> dict[str, Any]:
    """Evaluate exact spans, per-patient HPO codes and end-to-end span+code.

    Span matching uses exact half-open character offsets.  HPO code metrics use
    a set per patient, so repeated mentions of one code do not inflate results.
    End-to-end metrics require both the exact span and HPO code to match.
    """

    if not cases:
        raise ValueError("At least one patient case is required")
    seen_patients: set[str] = set()
    gold_spans: Counter[Any] = Counter()
    predicted_spans: Counter[Any] = Counter()
    gold_codes: Counter[Any] = Counter()
    predicted_codes: Counter[Any] = Counter()
    gold_end_to_end: Counter[Any] = Counter()
    predicted_end_to_end: Counter[Any] = Counter()
    gold_assertions: Counter[Any] = Counter()
    predicted_assertions: Counter[Any] = Counter()
    relaxed_true_positive = 0
    gold_count = 0
    predicted_count = 0

    for case_index, raw_case in enumerate(cases):
        case = _as_mapping(raw_case)
        raw_patient_id = case.get(patient_id_key, case.get("case_id"))
        patient_id = "" if raw_patient_id is None else str(raw_patient_id).strip()
        if not patient_id:
            raise ValueError(f"Case {case_index} is missing {patient_id_key!r}")
        if patient_id in seen_patients:
            raise ValueError(f"Duplicate patient ID would bias metrics: {patient_id}")
        seen_patients.add(patient_id)

        gold_mentions = [_as_mapping(item) for item in _as_sequence(case.get(gold_key))]
        predicted_mentions = [
            _as_mapping(item) for item in _as_sequence(case.get(predictions_key))
        ]
        gold_count += len(gold_mentions)
        predicted_count += len(predicted_mentions)
        patient_gold_codes: set[str] = set()
        patient_predicted_codes: set[str] = set()
        patient_gold_spans: list[tuple[int, int]] = []
        patient_predicted_spans: list[tuple[int, int]] = []

        for mention in gold_mentions:
            span = _mention_span(mention)
            code = _mention_code(mention)
            gold_spans[(patient_id, *span)] += 1
            patient_gold_spans.append(span)
            assertion = mention.get("assertion")
            if assertion not in (None, ""):
                gold_assertions[(patient_id, *span, str(assertion).lower())] += 1
            if code is not None:
                patient_gold_codes.add(code)
                gold_end_to_end[(patient_id, *span, code)] += 1
        for mention in predicted_mentions:
            span = _mention_span(mention)
            code = _mention_code(mention)
            predicted_spans[(patient_id, *span)] += 1
            patient_predicted_spans.append(span)
            assertion = mention.get("assertion")
            if assertion not in (None, ""):
                predicted_assertions[(patient_id, *span, str(assertion).lower())] += 1
            if code is not None:
                patient_predicted_codes.add(code)
                predicted_end_to_end[(patient_id, *span, code)] += 1
        for code in patient_gold_codes:
            gold_codes[(patient_id, code)] += 1
        for code in patient_predicted_codes:
            predicted_codes[(patient_id, code)] += 1
        relaxed_true_positive += _maximum_overlap_matches(patient_gold_spans, patient_predicted_spans)

    assertion_labels = sorted(
        {key[-1] for key in gold_assertions} | {key[-1] for key in predicted_assertions}
    )
    assertion_by_label: dict[str, dict[str, Any]] = {}
    for label in assertion_labels:
        gold_for_label = Counter({key: count for key, count in gold_assertions.items() if key[-1] == label})
        predicted_for_label = Counter(
            {key: count for key, count in predicted_assertions.items() if key[-1] == label}
        )
        assertion_by_label[label] = _counter_metric(gold_for_label, predicted_for_label)
    macro_f1 = (
        sum(item["f1"] for item in assertion_by_label.values()) / len(assertion_by_label)
        if assertion_by_label
        else 0.0
    )

    return {
        "unit": "mention_micro_with_patient_namespace",
        "patient_count": len(cases),
        "gold_mention_count": gold_count,
        "predicted_mention_count": predicted_count,
        "exact_span": _counter_metric(gold_spans, predicted_spans),
        "relaxed_overlap_span": _prf(
            relaxed_true_positive,
            predicted_count - relaxed_true_positive,
            gold_count - relaxed_true_positive,
        ),
        "assertion_exact_span": {
            **_counter_metric(gold_assertions, predicted_assertions),
            "macro_f1": macro_f1,
            "by_label": assertion_by_label,
        },
        "hpo_code": _counter_metric(gold_codes, predicted_codes),
        "exact_span_and_hpo_code": _counter_metric(gold_end_to_end, predicted_end_to_end),
    }


__all__ = [
    "evaluate_agent1_mentions",
    "evaluate_agent2_rankings",
    "evaluate_agent3_rankings",
    "binary_calibration_metrics",
    "patient_level_ranking_metrics",
]
