"""Agent 1: Vietnamese prenatal phenotype extraction and grounded HPO linking.

The extractor is intentionally deterministic.  It finds only phrases already
registered in :class:`HPOCatalog`, classifies their assertion in local clinical
context, extracts common prenatal modifiers, and links them to catalog-backed
HPO candidates.  It never generates an HPO identifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .hpo_catalog import HPOCatalog, normalize_text
from .schemas import (
    AssertionStatus,
    HPOCandidate,
    PatientProfile,
    PhenotypeMention,
    PhenotypeSubject,
)


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[,;.!?\n\r]|\b(?:nhưng|nhung|tuy\s+nhiên|tuy\s+nhien)\b",
    re.IGNORECASE,
)
_MEASUREMENT_RE = re.compile(
    r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(mm|cm|kg|g|bpm|%)\s*(?=$|\W)",
    re.IGNORECASE,
)

_ASSERTION_PATTERNS: tuple[
    tuple[AssertionStatus, tuple[str, ...]], ...
] = (
    # Precedence is clinically meaningful.  In particular, ``khong loai
    # tru`` must not be consumed by generic negation logic.
    (
        AssertionStatus.NOT_ASSESSED,
        (
            "han che danh gia",
            "khong danh gia duoc",
            "chua danh gia",
            "kho danh gia",
            "khong khao sat duoc",
            "chua khao sat",
            "khong quan sat duoc",
            "khong thay ro",
        ),
    ),
    (
        AssertionStatus.SUSPECTED,
        (
            "khong loai tru",
            "chua loai tru",
            "nghi ngo",
            "nghi",
            "goi y",
            "co kha nang",
            "co the",
            "theo doi",
        ),
    ),
    (
        AssertionStatus.ABSENT,
        (
            "chua ghi nhan",
            "khong ghi nhan",
            "khong phat hien",
            "khong thay",
            "khong co",
            "loai tru",
            "am tinh",
        ),
    ),
)

_STATUS_CERTAINTY = {
    AssertionStatus.PRESENT: 1.0,
    AssertionStatus.SUSPECTED: 0.5,
    AssertionStatus.ABSENT: 1.0,
    AssertionStatus.NOT_ASSESSED: 0.0,
}

_METHOD_PATTERNS = (
    ("ultrasound", ("sieu am", "ultrasound", "sonography")),
    ("mri", ("mri", "cong huong tu")),
    ("ct", ("ct scan", "chup ct", "cat lop vi tinh")),
    ("clinical_exam", ("kham lam sang", "tham kham")),
)

# Small reporting words may appear between otherwise exact lexicon tokens.
# They are skipped only when the trie cannot consume them, so ``khong co the
# chai`` still matches an alias that explicitly contains ``co``.
_OPTIONAL_REPORTING_TOKENS = frozenset({"co", "duoc"})

# Known source-document hallucination.  This guard is deliberately narrow: it
# prevents a poisoned alias table from mapping a posterior-fossa phrase to the
# catalog term Hemiplegia while still allowing specific posterior-fossa terms.
_KNOWN_UNSAFE_BINDINGS = frozenset(
    {
        ("posterior_fossa", "HP:0002301"),
        ("ventricular_dilation", "HP:0002280"),
        ("ventricular_dilation", "HP:0007272"),
    }
)


@dataclass(slots=True, frozen=True)
class _Token:
    normalized: str
    start: int
    end: int


@dataclass(slots=True, frozen=True)
class _PhraseMatch:
    start_token: int
    end_token: int
    normalized_alias: str
    alias_ids: frozenset[str]


class Agent1NLP:
    """Rule-based clinical mention extractor with catalog-grounded retrieval."""

    def __init__(
        self,
        catalog: HPOCatalog,
        *,
        top_k: int = 5,
        abstain_below: float = 0.55,
        review_below: float = 0.85,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= abstain_below <= 1.0:
            raise ValueError("abstain_below must be in [0, 1]")
        if not abstain_below <= review_below <= 1.0:
            raise ValueError("review_below must be in [abstain_below, 1]")
        self.catalog = catalog
        self.top_k = int(top_k)
        self.abstain_below = float(abstain_below)
        self.review_below = float(review_below)
        self._phrase_trie: dict[str, Any] = {}
        self._build_phrase_trie()

    @classmethod
    def from_files(
        cls,
        hpo_csv: str | Path,
        lexicon_csv: str | Path,
        *,
        ontology_version: str | None = None,
        top_k: int = 5,
        abstain_below: float = 0.55,
        review_below: float = 0.85,
        allow_unreviewed_lexicon: bool = False,
    ) -> "Agent1NLP":
        catalog = HPOCatalog.from_csv(
            hpo_csv,
            version=ontology_version,
            lexicon_path=lexicon_csv,
            allow_unreviewed_lexicon=allow_unreviewed_lexicon,
        )
        return cls(
            catalog,
            top_k=top_k,
            abstain_below=abstain_below,
            review_below=review_below,
        )

    def _build_phrase_trie(self) -> None:
        for normalized_alias, hpo_ids in self.catalog.alias_to_ids.items():
            alias_tokens = tuple(normalized_alias.split())
            if not alias_tokens:
                continue
            node = self._phrase_trie
            for token in alias_tokens:
                node = node.setdefault(token, {})
            node.setdefault("_aliases", {})[normalized_alias] = frozenset(hpo_ids)

    @staticmethod
    def _tokens(text: str) -> list[_Token]:
        tokens: list[_Token] = []
        for match in _TOKEN_RE.finditer(text):
            normalized = normalize_text(match.group(0))
            if normalized:
                tokens.append(_Token(normalized, match.start(), match.end()))
        return tokens

    def _longest_match(self, tokens: list[_Token], start: int) -> _PhraseMatch | None:
        node = self._phrase_trie
        index = start
        depth = 0
        best: tuple[tuple[int, int], _PhraseMatch] | None = None
        skipped = 0

        while index < len(tokens):
            token = tokens[index].normalized
            child = node.get(token)
            if child is not None:
                node = child
                index += 1
                depth += 1
                aliases = node.get("_aliases", {})
                for alias, hpo_ids in aliases.items():
                    candidate = _PhraseMatch(start, index, alias, hpo_ids)
                    key = (depth, index - start)
                    if best is None or key > best[0]:
                        best = (key, candidate)
                continue

            if token in _OPTIONAL_REPORTING_TOKENS and skipped < 2 and depth > 0:
                skipped += 1
                index += 1
                continue
            break

        return None if best is None else best[1]

    def _phrase_matches(self, text: str) -> tuple[list[_Token], list[_PhraseMatch]]:
        tokens = self._tokens(text)
        matches: list[_PhraseMatch] = []
        index = 0
        while index < len(tokens):
            match = self._longest_match(tokens, index)
            if match is None:
                index += 1
                continue
            matches.append(match)
            index = match.end_token
        return tokens, matches

    @staticmethod
    def _binding_kind(phrase: str) -> str | None:
        normalized = normalize_text(phrase)
        if "ho sau" in normalized or "posterior fossa" in normalized:
            return "posterior_fossa"
        if "nao that" in normalized or "ventricul" in normalized:
            return "ventricular_dilation"
        return None

    def validate_assignment(self, hpo_id: str, hpo_name: str) -> bool:
        """Return true only for an exact identifier-label pair in the catalog."""
        term = self.catalog.terms.get(hpo_id)
        return bool(
            term
            and term.name == hpo_name
            and not term.name.casefold().startswith("obsolete")
        )

    def retrieve(
        self,
        phrase: str,
        *,
        top_k: int | None = None,
        allowed_ids: Iterable[str] | None = None,
    ) -> list[HPOCandidate]:
        """Retrieve and validate candidates without permitting generated IDs."""
        limit = self.top_k if top_k is None else int(top_k)
        if limit < 1:
            return []
        allowed = None if allowed_ids is None else set(allowed_ids)
        binding_kind = self._binding_kind(phrase)
        grounded: list[HPOCandidate] = []
        seen: set[str] = set()
        # Ask for extra rows because invalid/obsolete candidates are filtered.
        for candidate in self.catalog.retrieve(phrase, top_k=max(limit * 3, limit)):
            if candidate.hpo_id in seen:
                continue
            if allowed is not None and candidate.hpo_id not in allowed:
                continue
            if not self.validate_assignment(candidate.hpo_id, candidate.hpo_name):
                continue
            if binding_kind and (binding_kind, candidate.hpo_id) in _KNOWN_UNSAFE_BINDINGS:
                continue
            grounded.append(candidate)
            seen.add(candidate.hpo_id)
            if len(grounded) >= limit:
                break
        return grounded

    @staticmethod
    def _clause(text: str, start: int, end: int) -> tuple[str, int, int]:
        left = 0
        for boundary in _CLAUSE_BOUNDARY_RE.finditer(text, 0, start):
            left = boundary.end()
        next_boundary = _CLAUSE_BOUNDARY_RE.search(text, end)
        right = len(text) if next_boundary is None else next_boundary.start()
        return text[left:right], left, right

    @staticmethod
    def _assertion(clause: str) -> tuple[AssertionStatus, str | None]:
        normalized = normalize_text(clause)
        padded = f" {normalized} "
        matches: list[tuple[tuple[int, int, int], AssertionStatus, str]] = []
        for priority, (status, patterns) in enumerate(_ASSERTION_PATTERNS):
            for pattern in patterns:
                if f" {pattern} " in padded:
                    # Prefer the most specific trigger.  This preserves
                    # ``khong loai tru`` over ``loai tru`` and prevents the
                    # substring ``co the`` inside ``khong co the chai`` from
                    # changing a negative phrase into SUSPECTED.
                    specificity = (len(pattern.split()), len(pattern), -priority)
                    matches.append((specificity, status, pattern))
        if matches:
            _, status, pattern = max(matches, key=lambda item: item[0])
            return status, pattern
        return AssertionStatus.PRESENT, None

    @staticmethod
    def _gestational_age(text: str) -> float | None:
        normalized = normalize_text(text)
        patterns = (
            re.compile(r"(?<!\d)(\d{1,2})\s*tuan(?:\s*(\d)\s*ngay)?\b"),
            re.compile(r"(?<!\d)(\d{1,2})\s*w(?:\s*\+?\s*(\d)\s*d?)?\b"),
        )
        for pattern in patterns:
            for match in pattern.finditer(normalized):
                weeks = int(match.group(1))
                days = int(match.group(2) or 0)
                if 10 <= weeks <= 45 and 0 <= days <= 6:
                    return round(weeks + days / 7.0, 3)
        return None

    @staticmethod
    def _method(clause: str, full_text: str, override: str | None) -> str | None:
        if override:
            return override.strip().casefold().replace(" ", "_")
        for scope in (normalize_text(clause), normalize_text(full_text)):
            hits = [
                method
                for method, patterns in _METHOD_PATTERNS
                if any(pattern in scope for pattern in patterns)
            ]
            if len(hits) == 1:
                return hits[0]
        return None

    @staticmethod
    def _laterality(clause: str) -> str | None:
        normalized = normalize_text(clause)
        if any(value in normalized for value in ("hai ben", "ca hai ben", "song phuong", "bilateral")):
            return "bilateral"
        if any(value in normalized for value in ("ben trai", "left sided", "left")):
            return "left"
        if any(value in normalized for value in ("ben phai", "right sided", "right")):
            return "right"
        if any(value in normalized for value in ("mot ben", "unilateral")):
            return "unilateral"
        return None

    @staticmethod
    def _severity(clause: str) -> str | None:
        normalized = normalize_text(clause)
        if any(value in normalized for value in ("rat nang", "muc do nang", "severe")):
            return "severe"
        if any(value in normalized for value in ("muc do vua", "trung binh", "moderate")):
            return "moderate"
        if any(value in normalized for value in ("muc do nhe", "nhe", "mild")):
            return "mild"
        return None

    @staticmethod
    def _distribution(clause: str) -> str | None:
        normalized = normalize_text(clause)
        if "lan toa" in normalized or "diffuse" in normalized:
            return "diffuse"
        if "khu tru" in normalized or "focal" in normalized:
            return "focal"
        return None

    @staticmethod
    def _progression(clause: str) -> str | None:
        normalized = normalize_text(clause)
        if "tien trien" in normalized or "progressive" in normalized:
            return "progressive"
        if "on dinh" in normalized or "stable" in normalized:
            return "stable"
        return None

    @staticmethod
    def _subject(clause: str) -> tuple[PhenotypeSubject, str]:
        """Assign a conservative subject using local prenatal-report context."""

        normalized = normalize_text(clause)
        padded = f" {normalized} "

        def has_phrase(marker: str) -> bool:
            return f" {marker} " in padded

        family_markers = (
            "tien su gia dinh",
            "con truoc",
            "anh chi em",
            "bo cua thai",
            "nguoi than",
            "family history",
        )
        explicit_fetal_markers = (
            "thai nhi",
            "fetal",
            "bao thai",
            "em be",
        )
        generic_fetal_markers = (
            "thai co",
            "thai ghi nhan",
            "tren thai",
            "o thai",
        )
        maternal_markers = (
            "thai phu",
            "san phu",
            "nguoi me",
            "me bi",
            "me co",
            "maternal",
        )
        if any(has_phrase(marker) for marker in family_markers):
            return PhenotypeSubject.FAMILY_HISTORY, "family_context_rule"
        if any(has_phrase(marker) for marker in explicit_fetal_markers):
            return PhenotypeSubject.FETUS, "explicit_fetal_context_rule"
        if any(has_phrase(marker) for marker in maternal_markers):
            return PhenotypeSubject.MOTHER, "maternal_context_rule"
        if any(has_phrase(marker) for marker in generic_fetal_markers):
            return PhenotypeSubject.FETUS, "fetal_context_rule"
        # This extractor is intentionally scoped to prenatal fetal phenotype
        # reports. The default remains reviewable and is recorded in provenance.
        return PhenotypeSubject.FETUS, "prenatal_fetus_default"

    @staticmethod
    def _measurements(clause: str) -> dict[str, float | str]:
        keys = {
            "mm": "width_mm",
            "cm": "length_cm",
            "g": "weight_g",
            "kg": "weight_kg",
            "bpm": "heart_rate_bpm",
            "%": "percent",
        }
        values: dict[str, float | str] = {}
        for match in _MEASUREMENT_RE.finditer(clause):
            value = float(match.group(1).replace(",", "."))
            unit = match.group(2).casefold()
            key = keys[unit]
            if key not in values:
                values[key] = int(value) if value.is_integer() else value
        return values

    def extract_mentions(
        self,
        text: str,
        *,
        observation_method: str | None = None,
    ) -> list[PhenotypeMention]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens, matches = self._phrase_matches(text)
        gestational_age = self._gestational_age(text)
        mentions: list[PhenotypeMention] = []

        for match in matches:
            start = tokens[match.start_token].start
            end = tokens[match.end_token - 1].end
            mention_text = text[start:end]
            clause, clause_start, clause_end = self._clause(text, start, end)
            assertion, assertion_trigger = self._assertion(clause)
            subject, subject_source = self._subject(clause)
            # Request enough rows to detect an exact alias attached to multiple
            # HPO terms even when callers expose top_k=1.
            selection_candidates = self.retrieve(
                match.normalized_alias,
                top_k=max(self.top_k, len(match.alias_ids), 2),
                allowed_ids=match.alias_ids,
            )
            top_score = selection_candidates[0].score if selection_candidates else 0.0
            ambiguous = bool(
                len(selection_candidates) > 1
                and abs(selection_candidates[0].score - selection_candidates[1].score) < 0.05
            )
            abstained = not selection_candidates or top_score < self.abstain_below or ambiguous
            selected = None if abstained else selection_candidates[0]
            candidates = selection_candidates[: self.top_k]
            needs_review = bool(
                abstained
                or top_score < self.review_below
                or assertion is not AssertionStatus.PRESENT
            )

            mentions.append(
                PhenotypeMention(
                    mention_text=mention_text,
                    span_start=start,
                    span_end=end,
                    normalized_phrase=match.normalized_alias,
                    hpo_id=None if selected is None else selected.hpo_id,
                    hpo_name=None if selected is None else selected.hpo_name,
                    assertion=assertion,
                    certainty=_STATUS_CERTAINTY[assertion],
                    gestational_age_weeks=gestational_age,
                    observation_method=self._method(clause, text, observation_method),
                    laterality=self._laterality(clause),
                    severity=self._severity(clause),
                    distribution=self._distribution(clause),
                    progression=self._progression(clause),
                    measurements=self._measurements(clause),
                    candidates=candidates,
                    model_confidence=top_score,
                    needs_review=needs_review,
                    extraction_confidence=1.0,
                    linking_confidence=top_score,
                    subject=subject,
                    provenance={
                        "extractor": "phrase_lexicon_longest_match_v1",
                        "matched_alias": match.normalized_alias,
                        "assertion_trigger": assertion_trigger,
                        "clause_span": [clause_start, clause_end],
                        "ambiguous_top_candidate": ambiguous,
                        "link_status": "abstain" if abstained else "grounded",
                        "subject_source": subject_source,
                    },
                )
            )
        return mentions

    def extract(
        self,
        text: str,
        *,
        case_id: str = "case-unknown",
        report_id: str | None = None,
        observation_method: str | None = None,
        clinical_context: dict[str, Any] | None = None,
    ) -> PatientProfile:
        mentions = self.extract_mentions(text, observation_method=observation_method)
        grounded_count = sum(mention.hpo_id is not None for mention in mentions)
        if not mentions:
            status = "abstain"
            reason = "no_lexicon_phrase_match"
        elif grounded_count == 0:
            status = "abstain"
            reason = "no_grounded_hpo_candidate"
        elif any(mention.needs_review for mention in mentions):
            status = "needs_review"
            reason = "one_or_more_mentions_require_review"
        else:
            status = "ok"
            reason = None
        return PatientProfile(
            case_id=case_id,
            mentions=mentions,
            source_text=text,
            clinician_approved=False,
            ontology_version=self.catalog.version,
            report_id=report_id,
            clinical_context=dict(clinical_context or {}),
            provenance={
                "agent": "agent1_nlp",
                "status": status,
                "reason": reason,
                "grounded_mentions": grounded_count,
                "total_mentions": len(mentions),
                "diagnostic_claim": False,
            },
        )

    # Friendly aliases for notebook and pipeline callers.
    run = extract
    process = extract
    parse_report = extract


HPOAgent1NLP = Agent1NLP


__all__ = ["Agent1NLP", "HPOAgent1NLP"]
