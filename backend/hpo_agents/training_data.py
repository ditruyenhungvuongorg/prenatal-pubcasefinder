"""Dataset preparation helpers for the three research agents.

Agent 1 is trained to extract phenotype mentions and assertion attributes.  It
is intentionally *not* trained to invent HPO identifiers; identifiers are added
after inference by :class:`hpo_agents.hpo_catalog.HPOCatalog` retrieval.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hpo_catalog import HPOCatalog
from .schemas import AssertionStatus, PhenotypeSubject, validate_hpo_id


SYSTEM_PROMPT_AGENT1 = """Bạn trích xuất kiểu hình thai từ báo cáo tiếng Việt.
Chỉ trả JSON hợp lệ theo schema {"mentions": [...]}. Mỗi mention gồm:
mention_text, span_start, span_end, normalized_phrase, assertion và attributes.
assertion chỉ là present, suspected, absent hoặc not_assessed.
attributes.subject chỉ là fetus, mother, family_history hoặc unknown.
Nếu có, attributes.extraction_confidence phải nằm trong [0, 1].
Không biến dấu hiệu không được nhắc đến thành absent; not_assessed không phải absent.
Không được tự sinh mã HPO. Không suy diễn điều không có trong câu."""


def _exact_occurrences(text: str, mention: str) -> list[int]:
    starts: list[int] = []
    cursor = 0
    while True:
        start = text.find(mention, cursor)
        if start < 0:
            return starts
        starts.append(start)
        cursor = start + 1


def _resolve_entity_span(
    text: str,
    entity: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int, int]:
    mention = entity.get("mention") or entity.get("mention_text")
    if not isinstance(mention, str) or not mention.strip():
        raise ValueError(f"{label}.mention is empty")

    raw_start = entity.get("span_start")
    raw_end = entity.get("span_end")
    has_start = raw_start is not None
    has_end = raw_end is not None
    if has_start != has_end:
        raise ValueError(f"{label} must provide both span_start and span_end")

    if has_start:
        if (
            not isinstance(raw_start, int)
            or isinstance(raw_start, bool)
            or not isinstance(raw_end, int)
            or isinstance(raw_end, bool)
        ):
            raise ValueError(f"{label} span values must be integers")
        start, end = raw_start, raw_end
        if not 0 <= start < end <= len(text):
            raise ValueError(f"{label} span is outside text bounds")
        if text[start:end] != mention:
            raise ValueError(f"{label} span does not exactly match mention")
        return start, end

    occurrences = _exact_occurrences(text, mention)
    if not occurrences:
        raise ValueError(f"{label}.mention is not an exact substring of text")
    if len(occurrences) > 1:
        raise ValueError(
            f"{label} must provide an explicit span because mention occurs multiple times"
        )
    start = occurrences[0]
    return start, start + len(mention)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            rows.append(value)
    return rows


def validate_agent1_annotations(
    rows: Sequence[Mapping[str, Any]],
    catalog: HPOCatalog | None = None,
    *,
    require_reviewed: bool = False,
) -> list[str]:
    """Return all validation problems without changing the input rows."""

    errors: list[str] = []
    valid_assertions = {status.value for status in AssertionStatus}
    for row_index, row in enumerate(rows):
        text = row.get("text")
        entities = row.get("entities")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"row[{row_index}].text is empty")
            continue
        if require_reviewed and row.get("review_status") != "approved":
            errors.append(f"row[{row_index}] is not clinician-approved")
        if not isinstance(entities, list):
            errors.append(f"row[{row_index}].entities must be a list")
            continue
        valid_spans: list[tuple[int, int, int]] = []
        for entity_index, entity in enumerate(entities):
            prefix = f"row[{row_index}].entities[{entity_index}]"
            if not isinstance(entity, Mapping):
                errors.append(f"{prefix} must be an object")
                continue
            mention = entity.get("mention") or entity.get("mention_text")
            if not isinstance(mention, str) or not mention.strip():
                errors.append(f"{prefix}.mention is empty")
            else:
                try:
                    start, end = _resolve_entity_span(text, entity, label=prefix)
                except ValueError as exc:
                    errors.append(str(exc))
                else:
                    valid_spans.append((start, end, entity_index))
            assertion = str(entity.get("assertion", "")).lower()
            if assertion not in valid_assertions:
                errors.append(f"{prefix}.assertion is invalid: {assertion!r}")
            subject = entity.get("subject", PhenotypeSubject.FETUS.value)
            try:
                PhenotypeSubject.coerce(str(subject))
            except ValueError:
                errors.append(f"{prefix}.subject is invalid: {subject!r}")
            hpo_id = entity.get("hpo_id")
            if hpo_id and catalog is not None and not catalog.validate(str(hpo_id)):
                errors.append(f"{prefix}.hpo_id is absent/obsolete in the pinned catalog: {hpo_id}")

        active: tuple[int, int, int] | None = None
        for span in sorted(valid_spans):
            if active is not None and span[0] < active[1]:
                errors.append(
                    f"row[{row_index}].entities[{span[2]}] span overlaps "
                    f"entities[{active[2]}]"
                )
            if active is None or span[1] > active[1]:
                active = span
    return errors


def validate_case_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_real: bool = False,
) -> list[str]:
    """Validate patient-level Agent-2 labels and detect split leakage.

    ``require_real`` enables the provenance gates needed before probability
    calibration. Synthetic rows remain valid fixtures when it is false.
    """

    errors: list[str] = []
    seen_case_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    allowed_statuses = {status.value for status in AssertionStatus}
    for row_index, row in enumerate(rows):
        prefix = f"row[{row_index}]"
        case_id = str(row.get("case_id", "")).strip()
        patient_id = str(row.get("patient_id", "")).strip()
        split = str(row.get("split", "")).strip().lower()
        if not case_id:
            errors.append(f"{prefix}.case_id is empty")
        elif case_id in seen_case_ids:
            errors.append(f"{prefix}.case_id is duplicated: {case_id}")
        seen_case_ids.add(case_id)
        if not patient_id:
            errors.append(f"{prefix}.patient_id is empty")
        if split not in {"train", "validation", "test"}:
            errors.append(f"{prefix}.split is invalid: {split!r}")

        group_id = str(row.get("family_id") or patient_id).strip()
        if group_id and split in {"train", "validation", "test"}:
            previous = group_splits.setdefault(group_id, split)
            if previous != split:
                errors.append(
                    f"{prefix} leaks patient/family group {group_id!r} across {previous}/{split}"
                )

        observations = row.get("observations")
        if not isinstance(observations, list) or not observations:
            errors.append(f"{prefix}.observations must be a non-empty list")
        else:
            states_by_hpo: dict[str, str] = {}
            positive_fetal = 0
            for observation_index, observation in enumerate(observations):
                item_prefix = f"{prefix}.observations[{observation_index}]"
                if not isinstance(observation, Mapping):
                    errors.append(f"{item_prefix} must be an object")
                    continue
                hpo_id = str(observation.get("hpo_id", "")).strip()
                try:
                    validate_hpo_id(hpo_id)
                except ValueError:
                    errors.append(f"{item_prefix}.hpo_id is invalid: {hpo_id!r}")
                status = str(observation.get("status", "")).strip().lower()
                if status not in allowed_statuses:
                    errors.append(f"{item_prefix}.status is invalid: {status!r}")
                raw_subject = observation.get("subject", PhenotypeSubject.FETUS.value)
                try:
                    subject = PhenotypeSubject.coerce(str(raw_subject))
                except ValueError:
                    errors.append(f"{item_prefix}.subject is invalid: {raw_subject!r}")
                    subject = PhenotypeSubject.UNKNOWN
                if (
                    status in {AssertionStatus.PRESENT.value, AssertionStatus.SUSPECTED.value}
                    and subject is PhenotypeSubject.FETUS
                ):
                    positive_fetal += 1
                previous_state = states_by_hpo.setdefault(hpo_id, status)
                if previous_state != status:
                    errors.append(
                        f"{item_prefix} contradicts another observation for {hpo_id}: "
                        f"{previous_state}/{status}"
                    )
                elif sum(1 for item in observations[:observation_index] if isinstance(item, Mapping) and str(item.get("hpo_id", "")).strip() == hpo_id) > 0:
                    errors.append(f"{item_prefix} duplicates HPO observation {hpo_id}")
            if positive_fetal == 0:
                errors.append(f"{prefix} needs at least one fetal present/suspected HPO")

        gold = row.get("gold_disease_ids")
        if not isinstance(gold, list) or not gold or any(not str(item).strip() for item in gold):
            errors.append(f"{prefix}.gold_disease_ids must be a non-empty string list")
        synthetic = row.get("synthetic")
        if not isinstance(synthetic, bool):
            errors.append(f"{prefix}.synthetic must be boolean")
        if not str(row.get("reference_standard", "")).strip():
            errors.append(f"{prefix}.reference_standard is empty")
        if str(row.get("review_status", "")).strip().lower() != "approved":
            errors.append(f"{prefix}.review_status must be approved")
        if require_real:
            if synthetic is not False:
                errors.append(f"{prefix} is not approved real clinical data (synthetic must be false)")
            for field in ("cohort_id", "hpo_release"):
                if not str(row.get(field, "")).strip():
                    errors.append(f"{prefix}.{field} is required for real calibration")
    if require_real:
        available_splits = {str(row.get("split", "")).strip().lower() for row in rows}
        missing_splits = {"train", "validation", "test"} - available_splits
        if missing_splits:
            errors.append(
                "real calibration requires non-empty train/validation/test splits; missing "
                + ", ".join(sorted(missing_splits))
            )
    return errors


def _entity_target(text: str, entity: Mapping[str, Any]) -> dict[str, Any]:
    mention = str(entity.get("mention") or entity.get("mention_text") or "")
    start, end = _resolve_entity_span(text, entity, label="entity")
    excluded = {
        "hpo_id",
        "hpo_name",
        "mention",
        "mention_text",
        "span_start",
        "span_end",
        "assertion",
        "normalized_phrase",
    }
    attributes = {
        key: value
        for key, value in entity.items()
        if key not in excluded and value not in (None, "", {}, [])
    }
    return {
        "mention_text": mention,
        "span_start": start,
        "span_end": end,
        "normalized_phrase": entity.get("normalized_phrase") or mention,
        "assertion": str(entity.get("assertion", "present")).lower(),
        "attributes": attributes,
    }


def agent1_sft_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    text = str(row["text"])
    entities = row.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("entities must be a list")
    targets: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for entity_index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            raise ValueError(f"entities[{entity_index}] must be an object")
        item = _entity_target(text, entity)
        span = (int(item["span_start"]), int(item["span_end"]))
        if any(span[0] < other_end and other_start < span[1] for other_start, other_end in occupied):
            raise ValueError(f"entities[{entity_index}] span overlaps another entity")
        occupied.append(span)
        targets.append(item)
    target = {"mentions": targets}
    return [
        {"role": "system", "content": SYSTEM_PROMPT_AGENT1},
        {"role": "user", "content": text},
        {
            "role": "assistant",
            "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def build_agent1_sft_dataset(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Convert approved annotations into chat SFT records without HPO IDs."""

    output = []
    for index, row in enumerate(rows):
        stable_id = row.get("case_id") or row.get("report_id") or f"row-{index:06d}"
        output.append(
            {
                "case_id": str(stable_id),
                "messages": agent1_sft_messages(row),
                "source": row.get("source", "expert_annotation"),
                "synthetic": bool(row.get("synthetic", False)),
            }
        )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in output:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output


def grouped_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 3407,
) -> dict[str, list[Mapping[str, Any]]]:
    """Deterministic patient/report-level split to prevent phrase leakage."""

    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must leave non-empty train, validation, and test proportions")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = row.get("patient_id") or row.get("case_id") or row.get("report_id")
        if not group:
            digest = hashlib.sha256(str(row.get("text", index)).encode("utf-8")).hexdigest()[:16]
            group = f"anonymous-{digest}"
        groups[str(group)].append(row)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError(
            "grouped_split requires at least 3 unique patient/case/report groups; "
            f"found {len(keys)}"
        )
    random.Random(seed).shuffle(keys)
    n_total = len(keys)
    n_train = int(n_total * train_fraction)
    n_validation = int(n_total * validation_fraction)
    if n_total >= 3:
        n_train = max(1, min(n_total - 2, n_train))
        n_validation = max(1, min(n_total - n_train - 1, n_validation))
    train_keys = set(keys[:n_train])
    validation_keys = set(keys[n_train : n_train + n_validation])
    result: dict[str, list[Mapping[str, Any]]] = {"train": [], "validation": [], "test": []}
    for key, values in groups.items():
        split = "train" if key in train_keys else "validation" if key in validation_keys else "test"
        result[split].extend(values)
    return result


def candidate_training_rows(
    case_candidates: Iterable[tuple[str, Sequence[Any], Sequence[str]]],
) -> list[dict[str, Any]]:
    """Create Agent-2/3 binary rows from candidate lists and gold disease IDs."""

    output: list[dict[str, Any]] = []
    for case_id, candidates, gold_ids in case_candidates:
        gold = {str(value) for value in gold_ids}
        for candidate in candidates:
            if hasattr(candidate, "as_dict"):
                item = dict(candidate.as_dict())
            elif isinstance(candidate, Mapping):
                item = dict(candidate)
            else:
                raise TypeError("candidate must be a mapping or expose as_dict()")
            output.append(
                {
                    "case_id": str(case_id),
                    "candidate": item,
                    "label": int(str(item.get("disease_id")) in gold),
                }
            )
    return output
