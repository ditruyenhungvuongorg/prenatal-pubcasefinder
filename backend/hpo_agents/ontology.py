"""Small, dependency-light readers for HPO OBO and annotation snapshots."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .schemas import validate_hpo_id


#: Root of the ``Phenotypic abnormality`` sub-ontology.  Official
#: ``phenotype.hpoa`` releases do annotate a handful of diseases with this term
#: directly, so readers must accept it; it carries no disease-discriminating
#: information, so scoring and information content must ignore it.
PHENOTYPIC_ABNORMALITY_ROOT = "HP:0000118"

FREQUENCY_CODES = {
    "HP:0040280": 0.995, # Obligate (bounded midpoint for likelihood use)
    "HP:0040281": 0.90,  # Very frequent
    "HP:0040282": 0.545, # Frequent
    "HP:0040283": 0.17,  # Occasional
    "HP:0040284": 0.025, # Very rare
    "HP:0040285": 0.0,   # Excluded
}

FREQUENCY_CODE_LABELS = {
    "HP:0040280": "Obligate",
    "HP:0040281": "Very frequent",
    "HP:0040282": "Frequent",
    "HP:0040283": "Occasional",
    "HP:0040284": "Very rare",
    "HP:0040285": "Excluded",
}

FREQUENCY_LABELS = {
    "obligate": FREQUENCY_CODES["HP:0040280"],
    "very frequent": FREQUENCY_CODES["HP:0040281"],
    "frequent": FREQUENCY_CODES["HP:0040282"],
    "occasional": FREQUENCY_CODES["HP:0040283"],
    "very rare": FREQUENCY_CODES["HP:0040284"],
    "excluded": FREQUENCY_CODES["HP:0040285"],
}


@dataclass(slots=True)
class HPOTerm:
    hpo_id: str
    name: str
    parents: set[str] = field(default_factory=set)
    synonyms: set[str] = field(default_factory=set)
    definition: str = ""
    alt_ids: set[str] = field(default_factory=set)
    is_obsolete: bool = False
    replaced_by: str | None = None


class HPOOntology:
    def __init__(self, terms: dict[str, HPOTerm], version: str | None = None) -> None:
        self.terms = terms
        self.version = version
        self.alt_to_primary: dict[str, str] = {}
        for hpo_id, term in terms.items():
            for alt_id in term.alt_ids:
                self.alt_to_primary[alt_id] = hpo_id
        self._ancestor_cache: dict[str, frozenset[str]] = {}
        self._depth_cache: dict[str, int] = {}

    @classmethod
    def from_obo(cls, path: str | Path) -> "HPOOntology":
        path = Path(path)
        version: str | None = None
        blocks: list[list[str]] = []
        current: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n\r")
                if line.startswith("data-version:"):
                    version = line.split(":", 1)[1].strip()
                if line == "[Term]":
                    if current:
                        blocks.append(current)
                    current = []
                elif line.startswith("["):
                    if current:
                        blocks.append(current)
                    current = []
                elif current or line.startswith("id: HP:"):
                    current.append(line)
            if current:
                blocks.append(current)

        terms: dict[str, HPOTerm] = {}
        for block in blocks:
            values: dict[str, list[str]] = defaultdict(list)
            for line in block:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    values[key].append(value)
            if not values.get("id") or not values["id"][0].startswith("HP:"):
                continue
            hpo_id = values["id"][0]
            term = HPOTerm(hpo_id=hpo_id, name=(values.get("name") or [hpo_id])[0])
            term.parents = {v.split(" ! ", 1)[0] for v in values.get("is_a", []) if v.startswith("HP:")}
            term.alt_ids = {v for v in values.get("alt_id", []) if v.startswith("HP:")}
            term.synonyms = {
                match.group(1)
                for v in values.get("synonym", [])
                if (match := re.match(r'^"(.*?)"', v))
            }
            if values.get("def"):
                match = re.match(r'^"(.*?)"', values["def"][0])
                term.definition = match.group(1) if match else values["def"][0]
            term.is_obsolete = (values.get("is_obsolete") or ["false"])[0].lower() == "true"
            replacement = (values.get("replaced_by") or [None])[0]
            term.replaced_by = replacement if replacement and replacement.startswith("HP:") else None
            terms[hpo_id] = term
        return cls(terms=terms, version=version)

    def resolve_id(self, hpo_id: str) -> str | None:
        current = hpo_id
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            primary = self.alt_to_primary.get(current, current)
            term = self.terms.get(primary)
            if term is None:
                return None
            if not term.is_obsolete:
                return primary
            if not term.replaced_by:
                return None
            current = term.replaced_by
        return None

    def ancestors(self, hpo_id: str, include_self: bool = True) -> frozenset[str]:
        resolved = self.resolve_id(hpo_id)
        if not resolved:
            return frozenset()
        if resolved not in self._ancestor_cache:
            seen: set[str] = set()
            stack = [resolved]
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                term = self.terms.get(node)
                if term:
                    stack.extend(term.parents)
            self._ancestor_cache[resolved] = frozenset(seen)
        result = self._ancestor_cache[resolved]
        return result if include_self else frozenset(x for x in result if x != resolved)

    def depth(self, hpo_id: str) -> int:
        resolved = self.resolve_id(hpo_id)
        if not resolved:
            return 0
        if resolved in self._depth_cache:
            return self._depth_cache[resolved]
        parents = [parent for parent in self.terms[resolved].parents if parent in self.terms]
        depth = 0 if not parents else 1 + max(self.depth(parent) for parent in parents)
        self._depth_cache[resolved] = depth
        return depth

    def common_ancestors(self, left: str, right: str) -> set[str]:
        return set(self.ancestors(left)) & set(self.ancestors(right))

    def mica(self, left: str, right: str, information_content: dict[str, float]) -> tuple[str | None, float]:
        common = self.common_ancestors(left, right)
        if not common:
            return None, 0.0
        node = max(common, key=lambda item: information_content.get(item, 0.0))
        return node, float(information_content.get(node, 0.0))

    def is_related(self, left: str, right: str) -> bool:
        return bool(self.common_ancestors(left, right))


def parse_frequency_detail(value: str | None, default: float = 0.5) -> dict[str, Any]:
    """Parse an HPOA frequency field while preserving what was actually curated.

    Returns the representative ``probability`` plus the provenance needed to
    tell ``"7/13 reported patients"`` apart from the category ``"Frequent"``
    apart from ``"nobody recorded a frequency"``.  ``kind`` is one of
    ``ratio``, ``percent``, ``category``, ``numeric`` or ``missing``; ``missing``
    means the caller must apply an explicit policy instead of assuming the
    ``default`` carries any evidence.
    """

    detail: dict[str, Any] = {
        "raw": None if value is None else str(value),
        "kind": "missing",
        "probability": default,
        "numerator": None,
        "denominator": None,
        "category_term": None,
        "category_label": None,
    }
    if value is None:
        return detail
    text = str(value).strip()
    if not text:
        return detail

    code = text.upper().split()[0]
    if code in FREQUENCY_CODES:
        detail.update(
            kind="category",
            probability=FREQUENCY_CODES[code],
            category_term=code,
            category_label=FREQUENCY_CODE_LABELS.get(code),
        )
        return detail

    lowered = text.casefold()
    for label in sorted(FREQUENCY_LABELS, key=len, reverse=True):
        if label in lowered:
            detail.update(
                kind="category",
                probability=FREQUENCY_LABELS[label],
                category_term=next(
                    (
                        term
                        for term, name in FREQUENCY_CODE_LABELS.items()
                        if name.casefold() == label
                    ),
                    None,
                ),
                category_label=label,
            )
            return detail

    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if percent:
        detail.update(
            kind="percent",
            probability=min(1.0, max(0.0, float(percent.group(1)) / 100.0)),
        )
        return detail

    fraction = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*", text)
    if fraction:
        numerator, denominator = map(float, fraction.groups())
        if not denominator:
            return detail
        detail.update(
            kind="ratio",
            probability=min(1.0, max(0.0, numerator / denominator)),
            numerator=numerator,
            denominator=denominator,
        )
        return detail

    try:
        numeric = float(text)
    except ValueError:
        return detail
    if not math.isfinite(numeric):
        return detail
    detail.update(
        kind="numeric",
        probability=min(1.0, max(0.0, numeric if numeric <= 1 else numeric / 100.0)),
    )
    return detail


def parse_frequency(value: str | None, default: float = 0.5) -> float:
    """Representative ``P(HPO | disease)``; see :func:`parse_frequency_detail`."""

    return float(parse_frequency_detail(value, default=default)["probability"])


def load_hpoa(path: str | Path, phenotypic_only: bool = True) -> dict[str, dict[str, Any]]:
    """Load phenotype.hpoa into disease profiles without mutating the source."""
    profiles: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        # Official HPOA snapshots start with several metadata lines and then a
        # tabular header whose first field is prefixed by ``#``.  Skip every
        # comment except that header; otherwise DictReader can accidentally use
        # ``#description`` or ``#version`` as the schema.
        rows = []
        header_seen = False
        for line in handle:
            if not header_seen:
                first = line.lstrip("\ufeff").lower()
                if first.startswith("#database_id\t") or first.startswith("database_id\t"):
                    header_seen = True
                    rows.append(line.lstrip("\ufeff"))
                continue
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line)
        if not rows:
            raise ValueError(f"No HPOA table header found in {path}")
        reader = csv.DictReader(rows, delimiter="\t")
        if reader.fieldnames:
            reader.fieldnames = [name.lstrip("#").strip() for name in reader.fieldnames]
        for row in reader:
            row = {str(k).lstrip("#").strip(): (v or "").strip() for k, v in row.items()}
            disease_id = row.get("database_id") or row.get("DatabaseID") or row.get("databaseId")
            hpo_id = row.get("hpo_id") or row.get("HPO_ID") or row.get("HPOId")
            aspect = row.get("aspect") or row.get("Aspect")
            if not disease_id or not hpo_id or (phenotypic_only and aspect and aspect != "P"):
                continue
            try:
                validate_hpo_id(hpo_id)
            except ValueError:
                continue
            profile = profiles.setdefault(
                disease_id,
                {
                    "disease_id": disease_id,
                    "disease_name": row.get("disease_name") or row.get("DiseaseName") or disease_id,
                    "phenotypes": [],
                },
            )
            qualifier = (row.get("qualifier") or row.get("Qualifier") or "").upper()
            frequency_detail = parse_frequency_detail(
                row.get("frequency") or row.get("Frequency"), default=0.5
            )
            frequency = float(frequency_detail["probability"])
            if qualifier == "NOT":
                frequency = 0.0
            profile["phenotypes"].append(
                {
                    "hpo_id": hpo_id,
                    "frequency": frequency,
                    "excluded": qualifier == "NOT" or frequency == 0.0,
                    # Frequency provenance: ``frequency`` above is a single
                    # representative number, so keep what was curated next to it
                    # rather than letting a default masquerade as data.
                    "frequency_raw": frequency_detail["raw"],
                    "frequency_kind": frequency_detail["kind"],
                    "frequency_missing": frequency_detail["kind"] == "missing",
                    "frequency_numerator": frequency_detail["numerator"],
                    "frequency_denominator": frequency_detail["denominator"],
                    "frequency_category_term": frequency_detail["category_term"],
                    "frequency_category_label": frequency_detail["category_label"],
                    # The phenotype root is a legal annotation but carries no
                    # discriminating information; readers keep it for audit and
                    # scorers must skip it.
                    "is_phenotype_root": hpo_id == PHENOTYPIC_ABNORMALITY_ROOT,
                    "onset": row.get("onset") or row.get("Onset") or None,
                    "evidence": row.get("evidence") or row.get("Evidence") or None,
                    "reference": row.get("reference") or row.get("DB_Reference") or None,
                    "sex": row.get("sex") or row.get("Sex") or None,
                    "modifier": row.get("modifier") or row.get("Modifier") or None,
                    "aspect": aspect or None,
                    "biocuration": row.get("biocuration") or row.get("Biocuration") or None,
                }
            )
    return profiles


def information_content_from_profiles(
    profiles: dict[str, dict[str, Any]], ontology: HPOOntology | None = None
) -> dict[str, float]:
    disease_sets: dict[str, set[str]] = defaultdict(set)
    for disease_id, profile in profiles.items():
        for item in profile.get("phenotypes", []):
            if item.get("excluded"):
                continue
            hpo_id = item["hpo_id"]
            if hpo_id == PHENOTYPIC_ABNORMALITY_ROOT:
                # Six official annotations point straight at the root.  Counting
                # them would make the least specific term in the ontology look
                # rare, and therefore informative, which it is not.
                continue
            nodes: Iterable[str] = ontology.ancestors(hpo_id) if ontology else (hpo_id,)
            for node in nodes:
                disease_sets[node].add(disease_id)
    total = max(1, len(profiles))
    return {node: -math.log((len(diseases) + 1.0) / (total + 1.0)) for node, diseases in disease_sets.items()}


def load_disease_gene_tsv(path: str | Path) -> dict[str, set[str]]:
    """Read common HPO genes_to_disease layouts into disease -> gene symbols."""
    result: dict[str, set[str]] = defaultdict(set)
    for row in load_disease_gene_rows(path):
        result[row["disease_id"]].add(row["gene_symbol"])
    return dict(result)


def load_disease_gene_rows(
    path: str | Path,
    *,
    source_version: str | None = None,
) -> list[dict[str, Any]]:
    """Read disease-gene rows while preserving provenance fields.

    A laboratory test modality is never inferred from the association type. It
    is propagated only when the input has an explicit ``test_evidence_type``
    column that was curated for this pipeline.
    """

    result: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            return []
        normalized = {name: re.sub(r"[^a-z0-9]", "", name.lower()) for name in reader.fieldnames}
        disease_col = next((k for k, v in normalized.items() if v in {"diseaseid", "databaseid"}), None)
        gene_col = next((k for k, v in normalized.items() if v in {"genesymbol", "symbol"}), None)
        if not disease_col or not gene_col:
            raise ValueError(f"Unsupported disease-gene columns: {reader.fieldnames}")
        for row in reader:
            disease_id = (row.get(disease_col) or "").strip()
            gene_symbol = (row.get(gene_col) or "").strip().upper()
            if disease_id and gene_symbol:
                by_normalized = {
                    normalized[column]: (row.get(column) or "").strip()
                    for column in reader.fieldnames
                }
                result.append(
                    {
                        "disease_id": disease_id,
                        "gene_symbol": gene_symbol,
                        "association_id": by_normalized.get("associationid") or None,
                        "evidence_type": by_normalized.get("evidencetype") or "curated",
                        "association_type": by_normalized.get("associationtype") or None,
                        "inheritance": by_normalized.get("inheritance") or None,
                        "test_evidence_type": by_normalized.get("testevidencetype") or None,
                        "source": by_normalized.get("source") or "HPO genes_to_disease",
                        "source_version": by_normalized.get("sourceversion") or source_version,
                    }
                )
    return result
