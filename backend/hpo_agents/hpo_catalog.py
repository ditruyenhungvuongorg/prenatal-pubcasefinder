"""Versioned HPO term catalog and grounded candidate retrieval."""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .ontology import HPOOntology
from .schemas import HPOCandidate, validate_hpo_id


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold().replace("đ", "d")
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9:+]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(slots=True, frozen=True)
class CatalogTerm:
    hpo_id: str
    name: str
    aliases: tuple[str, ...] = ()
    definition: str = ""


class HPOCatalog:
    def __init__(self, terms: Iterable[CatalogTerm], version: str | None = None) -> None:
        self.version = version
        self.terms: dict[str, CatalogTerm] = {}
        self.alias_to_ids: dict[str, set[str]] = defaultdict(set)
        self.explicit_alias_to_ids: dict[str, set[str]] = defaultdict(set)
        # Only aliases added explicitly through ``add_alias`` are eligible for
        # fuzzy retrieval.  Ontology names/synonyms remain available for exact
        # lookup, but mixing them into the same character index as a Vietnamese
        # lexicon can otherwise create cross-language false positives.
        self._alias_rows: list[tuple[str, str, str]] = []
        self._fuzzy_alias_keys: set[tuple[str, str]] = set()
        for term in terms:
            validate_hpo_id(term.hpo_id)
            if term.hpo_id in self.terms:
                continue
            self.terms[term.hpo_id] = term
            for alias in (term.name, *term.aliases):
                normalized = normalize_text(alias)
                if normalized:
                    self.alias_to_ids[normalized].add(term.hpo_id)
        self._vectorizer = None
        self._matrix = None

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        version: str | None = None,
        lexicon_path: str | Path | None = None,
        include_obsolete: bool = False,
        allow_unreviewed_lexicon: bool = False,
    ) -> "HPOCatalog":
        source_path = Path(path)
        if version is None:
            version = "csv-sha256:" + hashlib.sha256(source_path.read_bytes()).hexdigest()
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("HPO CSV has no header")
            normalized = {name: normalize_text(name).replace(" ", "") for name in reader.fieldnames}
            id_col = next((k for k, v in normalized.items() if v in {"mahpo", "hpoid", "id"}), None)
            name_col = next(
                (k for k, v in normalized.items() if v in {"tentrieuchungmota", "hponame", "name", "label"}),
                None,
            )
            if not id_col or not name_col:
                raise ValueError(f"Cannot identify HPO id/name columns: {reader.fieldnames}")
            terms = []
            for row in reader:
                hpo_id = (row.get(id_col) or "").strip().replace("_", ":")
                name = (row.get(name_col) or "").strip()
                try:
                    validate_hpo_id(hpo_id)
                except ValueError:
                    continue
                if name and (include_obsolete or not name.casefold().startswith("obsolete")):
                    terms.append(CatalogTerm(hpo_id=hpo_id, name=name))
        catalog = cls(terms=terms, version=version)
        if lexicon_path:
            catalog.add_lexicon_csv(
                lexicon_path,
                allow_unreviewed=allow_unreviewed_lexicon,
            )
        return catalog

    @classmethod
    def from_ontology(cls, ontology: HPOOntology) -> "HPOCatalog":
        return cls(
            (
                CatalogTerm(
                    hpo_id=term.hpo_id,
                    name=term.name,
                    aliases=tuple(sorted(term.synonyms)),
                    definition=term.definition,
                )
                for term in ontology.terms.values()
                if not term.is_obsolete
            ),
            version=ontology.version,
        )

    def add_alias(self, hpo_id: str, alias: str) -> None:
        """Register an explicit lexicon alias for exact and fuzzy retrieval.

        Ontology labels supplied to the constructor are exact-only.  Callers
        should use this method only for a curated language-specific alias (or
        explicitly accepted demo data).
        """

        validate_hpo_id(hpo_id)
        if hpo_id not in self.terms:
            raise KeyError(f"Alias targets HPO id absent from catalog: {hpo_id}")
        normalized = normalize_text(alias)
        if normalized:
            self.alias_to_ids[normalized].add(hpo_id)
            self.explicit_alias_to_ids[normalized].add(hpo_id)
            fuzzy_key = (normalized, hpo_id)
            if fuzzy_key not in self._fuzzy_alias_keys:
                self._fuzzy_alias_keys.add(fuzzy_key)
                self._alias_rows.append((normalized, alias, hpo_id))
                self._vectorizer = None
                self._matrix = None

    def add_lexicon_csv(
        self,
        path: str | Path,
        *,
        allow_unreviewed: bool = False,
    ) -> None:
        """Load aliases atomically, rejecting non-approved rows by default.

        ``allow_unreviewed=True`` is an explicit demo/research escape hatch.
        Production callers must leave it disabled so candidate or seed rows do
        not silently enter the grounding index.
        """

        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"hpo_id", "root_phrase_vi", "synonyms_vi"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Lexicon requires columns {sorted(required)}")
            rows = list(reader)

        pending: list[tuple[str, list[str]]] = []
        unapproved_lines: list[int] = []
        for row_index, row in enumerate(rows, start=2):
            review_status = str(row.get("review_status") or "").strip().casefold()
            if review_status != "approved":
                unapproved_lines.append(row_index)
            hpo_id = str(row.get("hpo_id") or "").strip()
            validate_hpo_id(hpo_id)
            if hpo_id not in self.terms:
                raise KeyError(
                    f"Lexicon line {row_index} targets HPO id absent from catalog: {hpo_id}"
                )
            aliases = [
                str(row.get("root_phrase_vi") or "").strip(),
                *(value.strip() for value in str(row.get("synonyms_vi") or "").split("|")),
            ]
            aliases = [alias for alias in aliases if alias]
            if not aliases:
                raise ValueError(f"Lexicon line {row_index} has no usable alias")
            pending.append((hpo_id, aliases))

        if unapproved_lines and not allow_unreviewed:
            lines = ", ".join(str(value) for value in unapproved_lines[:10])
            suffix = "..." if len(unapproved_lines) > 10 else ""
            raise ValueError(
                "Lexicon contains rows without review_status='approved' at lines "
                f"{lines}{suffix}; pass allow_unreviewed=True only for an explicit demo"
            )

        for hpo_id, aliases in pending:
            for alias in aliases:
                self.add_alias(hpo_id, alias)

    def _ensure_tfidf(self) -> None:
        if self._matrix is not None:
            return
        if not self._alias_rows:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform([row[0] for row in self._alias_rows])

    def retrieve(
        self,
        phrase: str,
        top_k: int = 5,
        *,
        allow_fuzzy: bool = False,
        explicit_only: bool = False,
    ) -> list[HPOCandidate]:
        """Retrieve exact catalog aliases or fuzzy explicit-lexicon aliases.

        Fuzzy matching never scans raw ontology labels/synonyms.  This prevents
        a Vietnamese phrase from being linked to an unrelated English label on
        character similarity alone.  Fuzzy retrieval is additionally opt-in;
        exact normalized aliases are the fail-closed production default.  With
        ``explicit_only=True``, even exact matches must come from ``add_alias``
        or an explicitly loaded lexicon rather than a raw ontology label.
        """

        query = normalize_text(phrase)
        if not query or top_k < 1:
            return []
        exact_index = self.explicit_alias_to_ids if explicit_only else self.alias_to_ids
        exact_ids = exact_index.get(query, set())
        if exact_ids:
            return [
                HPOCandidate(
                    hpo_id=hpo_id,
                    hpo_name=self.terms[hpo_id].name,
                    score=1.0,
                    matched_alias=phrase,
                    retrieval_method="exact_lexicon",
                )
                for hpo_id in sorted(exact_ids)[:top_k]
            ]

        if not allow_fuzzy:
            return []

        self._ensure_tfidf()
        scored: dict[str, tuple[float, str]] = {}
        if self._matrix is not None and self._vectorizer is not None:
            vector = self._vectorizer.transform([query])
            similarities = (self._matrix @ vector.T).toarray().ravel()
            candidate_count = min(top_k * 5, len(similarities))
            if candidate_count == 0:
                return []
            if candidate_count == len(similarities):
                candidate_rows = np.arange(len(similarities))
            else:
                candidate_rows = np.argpartition(similarities, -candidate_count)[-candidate_count:]
            for index in candidate_rows:
                score = float(similarities[index])
                normalized_alias, original_alias, hpo_id = self._alias_rows[int(index)]
                current = scored.get(hpo_id)
                if current is None or score > current[0]:
                    scored[hpo_id] = (score, original_alias)
        else:
            from difflib import SequenceMatcher

            for normalized_alias, original_alias, hpo_id in self._alias_rows:
                score = SequenceMatcher(None, query, normalized_alias).ratio()
                current = scored.get(hpo_id)
                if current is None or score > current[0]:
                    scored[hpo_id] = (score, original_alias)

        return [
            HPOCandidate(
                hpo_id=hpo_id,
                hpo_name=self.terms[hpo_id].name,
                score=score,
                matched_alias=alias,
                retrieval_method="char_tfidf" if self._matrix is not None else "sequence_match",
            )
            for hpo_id, (score, alias) in sorted(scored.items(), key=lambda item: item[1][0], reverse=True)[:top_k]
        ]

    def validate(self, hpo_id: str) -> bool:
        return hpo_id in self.terms
