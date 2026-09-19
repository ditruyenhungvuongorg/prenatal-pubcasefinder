"""Agent 2: phenotype-to-disease profile prioritisation.

The matcher is deliberately offline and read-only.  It reads ``phenotype.hpoa``
and, when available, ``hp.obo``; it never modifies either source.  The output is
a transparent ranking for expert review, not a diagnostic statement.

The implementation exposes three complementary coverage measures plus a
LIRICAL-inspired (but not a reimplementation of LIRICAL) log-likelihood ratio:

* ``exact_coverage`` - exact positive HPO matches.
* ``ancestor_coverage`` - exact or directly comparable ancestor/descendant
  matches in the HPO graph.
* ``ic_weighted_coverage`` - information-content-weighted semantic coverage.
* ``negative_conflict`` - explicitly absent findings that the disease profile
  expects to be present.

``rank`` orders profiles by ``RankingStrategy``: IC-weighted coverage first by
default, or the original likelihood-dominated ``score``.

Only Python's standard library is needed for ranking.  Calibration is optional:
XGBoost is tried when requested and installed, then scikit-learn logistic
regression, then a small NumPy logistic-regression fallback.
"""

from __future__ import annotations

import math
import re
import hashlib
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import disease_probability as _probability
from .agent2_schema import (
    PHENOTYPIC_ABNORMALITY_ROOT,
    Abstention,
    AbstentionReason,
    Agent2Warning,
    Agent2ContractError,
    BackgroundModel,
    CalibratorMetadata,
    ClinicalContext,
    DiseasePrior,
    DiseasePriorProvider,
    IdentityDiseaseResolver,
    MissingFrequencyPolicy,
    PhenotypeObservation,
    PhenotypeStatus,
    RankingStrategy,
    ReviewerStatus,
    StatusWeightPolicy,
    WarningCode,
    load_disease_xref,
    normalize_hpo_id as _normalize_hpo_id,
    parse_agent1_to_agent2_request,
)
from .disease_probability import (
    UNKNOWN_DISEASE_ID,
    UNKNOWN_DISEASE_NAME,
    LikelihoodConfig,
    PosteriorEstimate,
    SingleCauseEntry,
    UnknownClass,
)
from .ontology import parse_frequency, parse_frequency_detail


_HPO_ID_RE = re.compile(r"^HP:\d{7}$", re.IGNORECASE)
_EPSILON = 1e-9
_DEFAULT_DISEASE_FREQUENCY = 0.5
_MIN_PHENOTYPE_PROBABILITY = 1e-4

#: Bumped whenever the serialised output or feature vector changes shape.
AGENT2_SCHEMA_VERSION = "agent2-output/2"
FEATURE_VERSION = "agent2-features/1"

#: A positive finding below this information content is treated as too general
#: to rank on.  Deliberately permissive: it catches the phenotype root and terms
#: annotated on almost every disease, and nothing else.  Tighten it only with
#: cohort data that justifies a specific threshold.
_MIN_INFORMATIVE_IC = 0.05


def _dedupe_warnings(warnings: Sequence[Agent2Warning]) -> list[Agent2Warning]:
    """Keep the first warning of each code; repeated codes add no information."""

    seen: set[str] = set()
    unique: list[Agent2Warning] = []
    for item in warnings:
        if item.code.value in seen:
            continue
        seen.add(item.code.value)
        unique.append(item)
    return unique


# ``PhenotypeStatus`` and ``PhenotypeObservation`` now live in ``agent2_schema``
# so Agent 1, Agent 2 and reviewers share one definition.  They are re-exported
# here unchanged -- same names, same positional signature ``(hpo_id, status)``,
# same ``coerce`` helper -- so existing imports and notebooks keep working.


@dataclass(frozen=True)
class HPOAAnnotation:
    """Source HPOA row retained for audit and clinical review."""

    hpo_id: str
    qualifier: str
    frequency: float
    reference: Optional[str]
    evidence: Optional[str]
    onset: Optional[str]
    sex: Optional[str]
    modifier: Optional[str]
    aspect: Optional[str]
    biocuration: Optional[str]
    # ``frequency`` above is one representative number.  These keep the curated
    # source visible so a reviewer can tell "7/13 patients" apart from the
    # category "Frequent" apart from "no frequency was recorded".
    frequency_raw: Optional[str] = None
    frequency_kind: str = "missing"
    frequency_numerator: Optional[float] = None
    frequency_denominator: Optional[float] = None
    frequency_category_term: Optional[str] = None
    frequency_category_label: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "hpo_id": self.hpo_id,
            "qualifier": self.qualifier,
            "frequency": self.frequency,
            "reference": self.reference,
            "evidence": self.evidence,
            "onset": self.onset,
            "sex": self.sex,
            "modifier": self.modifier,
            "aspect": self.aspect,
            "biocuration": self.biocuration,
            "frequency_raw": self.frequency_raw,
            "frequency_kind": self.frequency_kind,
            "frequency_numerator": self.frequency_numerator,
            "frequency_denominator": self.frequency_denominator,
            "frequency_category_term": self.frequency_category_term,
            "frequency_category_label": self.frequency_category_label,
        }

    def frequency_detail(self) -> dict[str, Any]:
        """Re-expose the parsed frequency in ``parse_frequency_detail`` shape."""

        return {
            "raw": self.frequency_raw,
            "kind": self.frequency_kind,
            "probability": self.frequency,
            "numerator": self.frequency_numerator,
            "denominator": self.frequency_denominator,
            "category_term": self.frequency_category_term,
            "category_label": self.frequency_category_label,
        }


@dataclass(frozen=True)
class EvidenceItem:
    """Per-observation evidence used for one disease-profile score."""

    observed_hpo_id: str
    status: str
    matched_hpo_id: Optional[str]
    relation: str
    distance: Optional[int]
    ic_similarity: float
    disease_frequency: Optional[float]
    disease_probability: Optional[float]
    background_probability: Optional[float]
    log_likelihood_ratio: float
    weighted_contribution: float
    message: str
    hpoa_annotations: tuple[HPOAAnnotation, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_hpo_id": self.observed_hpo_id,
            "status": self.status,
            "matched_hpo_id": self.matched_hpo_id,
            "relation": self.relation,
            "distance": self.distance,
            "ic_similarity": self.ic_similarity,
            "disease_frequency": self.disease_frequency,
            "disease_probability": self.disease_probability,
            "background_probability": self.background_probability,
            "log_likelihood_ratio": self.log_likelihood_ratio,
            "weighted_contribution": self.weighted_contribution,
            "message": self.message,
            "hpoa_annotations": [item.as_dict() for item in self.hpoa_annotations],
        }


@dataclass(frozen=True)
class CandidateScore:
    """Transparent ranking result for one disease profile."""

    disease_id: str
    disease_name: str
    score: float
    compatibility_score: float
    confidence_score: Optional[float]
    exact_coverage: float
    ancestor_coverage: float
    ic_weighted_coverage: float
    negative_conflict: float
    lirical_log_likelihood: float
    calibration_method: Optional[str]
    evidence: tuple[EvidenceItem, ...]
    explanation: tuple[str, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    scope_note: str = (
        "Computational prioritisation only; review the evidence with qualified "
        "clinical-genetics professionals and confirm against appropriate testing."
    )

    def feature_vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in HPOAgent2Matcher.FEATURE_NAMES)

    @property
    def probability(self) -> Optional[float]:
        """Deprecated compatibility alias; this is not a disease posterior."""

        return self.confidence_score

    def as_dict(self) -> dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "disease_name": self.disease_name,
            "score": self.score,
            "compatibility_score": self.compatibility_score,
            "confidence_score": self.confidence_score,
            "metrics": {
                "exact_coverage": self.exact_coverage,
                "ancestor_coverage": self.ancestor_coverage,
                "ic_weighted_coverage": self.ic_weighted_coverage,
                "negative_conflict": self.negative_conflict,
                "lirical_log_likelihood": self.lirical_log_likelihood,
            },
            "calibration_method": self.calibration_method,
            "evidence": [item.as_dict() for item in self.evidence],
            "explanation": list(self.explanation),
            "provenance": dict(self.provenance),
            "confidence_semantics": (
                "Empirical ranking confidence from the configured calibrator; "
                "not a disease posterior probability."
                if self.confidence_score is not None
                else "No empirical confidence model was applied."
            ),
            # rank() has no disease prior, so it can only order candidates.
            # These labels make that explicit rather than leaving a reader to
            # infer that compatibility_score means a chance of having the
            # disease.  Use analyze() to obtain an actual posterior.
            "estimated_probability": None,
            "probability_kind": _probability.ProbabilityKind.RANKING_SCORE.value,
            "probability_scope": _probability.ProbabilityScope.NOT_APPLICABLE.value,
            "calibrated": False,
            "schema_version": AGENT2_SCHEMA_VERSION,
            "scope_note": self.scope_note,
        }


@dataclass(frozen=True)
class DiseaseCandidateOutput:
    """One ranked disease with its probability semantics made explicit.

    ``raw_score`` and ``compatibility_score`` order and describe the match.
    ``posterior`` is the only place an absolute probability may appear, and it
    always carries its own ``kind``, ``scope`` and ``calibrated`` flags, so a
    reader can never mistake a compatibility of 1.0 for "100% chance of this
    disease".
    """

    disease_id: str
    disease_name: str
    concept_id: str
    rank: int
    raw_score: float
    compatibility_score: float
    posterior: PosteriorEstimate
    likelihood: _probability.LogLikelihoodResult
    matched_evidence: tuple[EvidenceItem, ...]
    conflicting_evidence: tuple[EvidenceItem, ...]
    ignored_evidence: tuple[EvidenceItem, ...]
    warnings: tuple[Agent2Warning, ...] = ()
    candidate: Optional[CandidateScore] = None
    merged_disease_ids: tuple[str, ...] = ()

    @property
    def estimated_probability(self) -> Optional[float]:
        return self.posterior.probability

    @property
    def probability_kind(self) -> str:
        return self.posterior.kind.value

    @property
    def probability_scope(self) -> str:
        return self.posterior.scope.value

    @property
    def calibrated(self) -> bool:
        return self.posterior.calibrated

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "disease_id": self.disease_id,
            "disease_name": self.disease_name,
            "concept_id": self.concept_id,
            "merged_disease_ids": list(self.merged_disease_ids),
            "rank": self.rank,
            "raw_score": self.raw_score,
            "compatibility_score": self.compatibility_score,
            "compatibility_semantics": (
                "Bounded match quality between the observed phenotypes and this "
                "curated profile. It is not a probability of having the disease."
            ),
            "matched_evidence": [item.as_dict() for item in self.matched_evidence],
            "conflicting_evidence": [item.as_dict() for item in self.conflicting_evidence],
            "ignored_evidence": [item.as_dict() for item in self.ignored_evidence],
            "per_feature_contributions": [
                term.as_dict() for term in self.likelihood.terms
            ],
            "log_likelihood_ratio": self.likelihood.log_likelihood_ratio,
            "warnings": [item.as_dict() for item in self.warnings],
            "schema_version": AGENT2_SCHEMA_VERSION,
        }
        payload.update(self.posterior.as_dict())
        return payload


@dataclass(frozen=True)
class Agent2Result:
    """Full Agent 2 output: ranking, probability semantics, caveats, provenance."""

    candidates: tuple[DiseaseCandidateOutput, ...]
    observations: tuple[PhenotypeObservation, ...]
    warnings: tuple[Agent2Warning, ...] = ()
    abstention: Optional[Abstention] = None
    context: Optional[ClinicalContext] = None
    probability_mode: str = "none"
    unknown_class: Optional[Mapping[str, Any]] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    calibrator_metadata: Optional[Mapping[str, Any]] = None
    scope_note: str = (
        "Computational prioritisation for expert review only. Not a diagnosis, "
        "and not validated for clinical use."
    )

    @property
    def abstained(self) -> bool:
        return self.abstention is not None

    @property
    def top(self) -> Optional[DiseaseCandidateOutput]:
        return self.candidates[0] if self.candidates else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AGENT2_SCHEMA_VERSION,
            "abstained": self.abstained,
            "abstention": self.abstention.as_dict() if self.abstention else None,
            "probability_mode": self.probability_mode,
            "candidates": [item.as_dict() for item in self.candidates],
            "observations": [item.as_dict() for item in self.observations],
            "warnings": [item.as_dict() for item in self.warnings],
            "unknown_class": dict(self.unknown_class) if self.unknown_class else None,
            "context": self.context.as_dict() if self.context else None,
            "calibrator_metadata": (
                dict(self.calibrator_metadata) if self.calibrator_metadata else None
            ),
            "provenance": dict(self.provenance),
            "scope_note": self.scope_note,
        }


@dataclass
class _DiseaseProfile:
    disease_id: str
    disease_name: str
    positive_frequencies: dict[str, float]
    negative_terms: set[str]
    annotations: dict[str, list[HPOAAnnotation]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Match:
    term: str
    relation: str
    distance: Optional[int]
    ic_similarity: float
    strength: float
    frequency: float


@dataclass(frozen=True)
class _SanitationResult:
    """Scorable observations plus an account of what sanitation changed."""

    observations: tuple[PhenotypeObservation, ...]
    ignored_root_terms: tuple[str, ...] = ()
    merged_duplicate_terms: tuple[str, ...] = ()
    removed_redundant_terms: tuple[str, ...] = ()
    unknown_terms: tuple[str, ...] = ()


class HPOOntology:
    """Small read-only view of the parent relationships needed by the matcher."""

    def __init__(
        self,
        parents: Optional[Mapping[str, Iterable[str]]] = None,
        names: Optional[Mapping[str, str]] = None,
        aliases: Optional[Mapping[str, str]] = None,
        version: Optional[str] = None,
    ) -> None:
        self.parents = {
            _normalize_hpo_id(term): frozenset(_normalize_hpo_id(parent) for parent in values)
            for term, values in (parents or {}).items()
        }
        self.names = {_normalize_hpo_id(key): value for key, value in (names or {}).items()}
        self.aliases = {
            _normalize_hpo_id(alias): _normalize_hpo_id(target)
            for alias, target in (aliases or {}).items()
        }
        self.version = version
        self.known_terms = frozenset(
            set(self.parents)
            | {parent for values in self.parents.values() for parent in values}
            | set(self.names)
            | set(self.aliases.values())
        )
        self._ancestor_cache: dict[str, dict[str, int]] = {}

    @classmethod
    def from_obo(cls, path: str | Path) -> "HPOOntology":
        """Parse only ``id``, ``alt_id`` and ``is_a`` fields from an OBO file."""

        source = Path(path)
        parents: dict[str, set[str]] = defaultdict(set)
        names: dict[str, str] = {}
        aliases: dict[str, str] = {}
        stanza: dict[str, list[str]] = {}
        in_term = False
        version: Optional[str] = None

        def flush() -> None:
            nonlocal stanza
            if not in_term or "id" not in stanza:
                stanza = {}
                return
            term = _normalize_hpo_id(stanza["id"][0])
            if stanza.get("is_obsolete", ["false"])[0].strip().lower() == "true":
                replacement = stanza.get("replaced_by", [])
                if replacement:
                    aliases[term] = _normalize_hpo_id(replacement[0].split()[0])
                stanza = {}
                return
            if stanza.get("name"):
                names[term] = stanza["name"][0]
            for parent_value in stanza.get("is_a", []):
                parents[term].add(_normalize_hpo_id(parent_value.split()[0]))
            for alt_value in stanza.get("alt_id", []):
                aliases[_normalize_hpo_id(alt_value.split()[0])] = term
            parents.setdefault(term, set())
            stanza = {}

        with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if line.startswith("data-version:"):
                    version = line.split(":", 1)[1].strip() or None
                    continue
                if line == "[Term]":
                    flush()
                    in_term = True
                    continue
                if line.startswith("[") and line.endswith("]"):
                    flush()
                    in_term = False
                    continue
                if not in_term or not line or line.startswith("!") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                stanza.setdefault(key.strip(), []).append(value.strip())
        flush()
        return cls(parents=parents, names=names, aliases=aliases, version=version)

    def resolve(self, hpo_id: str) -> str:
        term = _normalize_hpo_id(hpo_id)
        seen: set[str] = set()
        while term in self.aliases and term not in seen:
            seen.add(term)
            term = self.aliases[term]
        return term

    def contains(self, hpo_id: str) -> bool:
        """Return whether an ID resolves to an active term in this snapshot."""

        return self.resolve(hpo_id) in self.known_terms

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        ancestor = self.resolve(ancestor)
        descendant = self.resolve(descendant)
        return ancestor != descendant and ancestor in self.ancestors_with_distance(descendant)

    def is_phenotype_root(self, hpo_id: str) -> bool:
        """Whether an ID *is* ``HP:0000118`` itself."""

        return self.resolve(hpo_id) == PHENOTYPIC_ABNORMALITY_ROOT

    def is_within_phenotypic_abnormality(self, hpo_id: str) -> bool:
        """Whether a term is the phenotype root **or** one of its descendants.

        This is the acceptance test for reading a snapshot.  Official
        ``phenotype.hpoa`` releases annotate a few diseases with ``HP:0000118``
        directly, and refusing to load them would discard those disease profiles
        entirely.  Terms from other branches (inheritance, clinical modifiers,
        the ``All`` root) are still rejected.
        """

        term = self.resolve(hpo_id)
        if PHENOTYPIC_ABNORMALITY_ROOT not in self.known_terms:
            return True
        return PHENOTYPIC_ABNORMALITY_ROOT in self.ancestors_with_distance(term)

    def is_phenotypic_abnormality(self, hpo_id: str) -> bool:
        """Whether a term can discriminate between diseases.

        The phenotype root is inside the branch but is true of every abnormal
        phenotype, so it is excluded here: this is the scoring test, not the
        loading test.  Use :meth:`is_within_phenotypic_abnormality` to decide
        whether a term is acceptable at all.
        """

        return self.is_within_phenotypic_abnormality(hpo_id) and not self.is_phenotype_root(hpo_id)

    def ancestors_with_distance(self, hpo_id: str) -> dict[str, int]:
        term = self.resolve(hpo_id)
        cached = self._ancestor_cache.get(term)
        if cached is not None:
            return dict(cached)
        distances = {term: 0}
        queue: deque[str] = deque([term])
        while queue:
            child = queue.popleft()
            next_distance = distances[child] + 1
            for parent in self.parents.get(child, ()):  # unknown terms simply have no parents
                if parent not in distances or next_distance < distances[parent]:
                    distances[parent] = next_distance
                    queue.append(parent)
        self._ancestor_cache[term] = distances
        return dict(distances)


class _NumpyLogisticCalibrator:
    """Dependency-light binary logistic regression used when sklearn is absent."""

    def __init__(self, *, iterations: int = 1500, learning_rate: float = 0.08, l2: float = 1e-3):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2 = l2
        self.mean_: Any = None
        self.scale_: Any = None
        self.coef_: Any = None
        self.intercept_: float = 0.0

    def fit(self, x: Any, y: Any) -> "_NumpyLogisticCalibrator":
        import numpy as np

        matrix = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float)
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        normalized = (matrix - self.mean_) / self.scale_
        self.coef_ = np.zeros(normalized.shape[1], dtype=float)
        self.intercept_ = 0.0
        for _ in range(self.iterations):
            logits = np.clip(normalized @ self.coef_ + self.intercept_, -35.0, 35.0)
            predictions = 1.0 / (1.0 + np.exp(-logits))
            residual = predictions - target
            gradient = normalized.T @ residual / len(target) + self.l2 * self.coef_
            intercept_gradient = float(residual.mean())
            self.coef_ -= self.learning_rate * gradient
            self.intercept_ -= self.learning_rate * intercept_gradient
        return self

    def predict_proba(self, x: Any) -> Any:
        import numpy as np

        matrix = np.asarray(x, dtype=float)
        normalized = (matrix - self.mean_) / self.scale_
        logits = np.clip(normalized @ self.coef_ + self.intercept_, -35.0, 35.0)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack((1.0 - positive, positive))


class HPOAgent2Matcher:
    """Rank HPOA disease profiles against assessed patient phenotypes."""

    FEATURE_NAMES = (
        "exact_coverage",
        "ancestor_coverage",
        "ic_weighted_coverage",
        "negative_conflict",
        "lirical_log_likelihood",
    )

    def __init__(
        self,
        profiles: Sequence[_DiseaseProfile],
        ontology: Optional[HPOOntology] = None,
        *,
        provenance: Optional[Mapping[str, Any]] = None,
        status_weight_policy: Optional[StatusWeightPolicy] = None,
        likelihood_config: Optional[LikelihoodConfig] = None,
        missing_frequency_policy: MissingFrequencyPolicy = MissingFrequencyPolicy.NEUTRAL,
        background_model: BackgroundModel = BackgroundModel.ANNOTATION_BACKGROUND,
        prior_provider: Optional[DiseasePriorProvider] = None,
        disease_resolver: Optional[Any] = None,
        ranking_strategy: RankingStrategy | str = RankingStrategy.IC_COVERAGE,
    ):
        if not profiles:
            raise ValueError("At least one HPOA disease profile is required")
        self.ranking_strategy = RankingStrategy(ranking_strategy)
        self.ontology = ontology or HPOOntology()
        # Root annotations are recorded before canonicalisation strips them from
        # the scoring maps, so the audit trail keeps what the snapshot said.
        self.root_annotated_diseases = tuple(
            sorted(
                profile.disease_id
                for profile in profiles
                if PHENOTYPIC_ABNORMALITY_ROOT
                in (*profile.positive_frequencies, *profile.negative_terms)
            )
        )
        self.profiles = tuple(self._canonicalize_profile(profile) for profile in profiles)
        self._profile_by_id = {profile.disease_id: profile for profile in self.profiles}
        self.provenance = dict(provenance or {})
        self.status_weight_policy = status_weight_policy or StatusWeightPolicy()
        self.likelihood_config = likelihood_config or _probability.DEFAULT_CONFIG
        self.missing_frequency_policy = MissingFrequencyPolicy(missing_frequency_policy)
        self.background_model = BackgroundModel(background_model)
        self.prior_provider = prior_provider
        self.disease_resolver = disease_resolver or IdentityDiseaseResolver()
        self._known_profile_terms = frozenset(
            term
            for profile in self.profiles
            for term in (*profile.positive_frequencies, *profile.negative_terms)
        )
        if self.ontology.known_terms:
            unknown_profile_terms = sorted(
                term for term in self._known_profile_terms if not self.ontology.contains(term)
            )
            if unknown_profile_terms:
                preview = ", ".join(unknown_profile_terms[:10])
                raise ValueError(
                    "HPOA/OBO snapshot mismatch: phenotype.hpoa contains term(s) "
                    f"absent from hp.obo: {preview}"
                )
            # Accept the phenotype root itself -- official releases annotate a
            # few diseases with it -- but keep rejecting other branches.
            invalid_profile_terms = sorted(
                term
                for term in self._known_profile_terms
                if not self.ontology.is_within_phenotypic_abnormality(term)
            )
            if invalid_profile_terms:
                preview = ", ".join(invalid_profile_terms[:10])
                raise ValueError(
                    "HPOA phenotype aspect contains term(s) outside HP:0000118: "
                    f"{preview}"
                )
        self._document_counts = self._build_document_counts()
        self._group_cache: dict[str, Optional[str]] = {}
        self._calibrator: Any = None
        self.calibration_method: Optional[str] = None
        self._calibrator_metadata: Optional[CalibratorMetadata] = None

    @classmethod
    def from_files(
        cls,
        phenotype_hpoa: str | Path,
        hp_obo: str | Path | None = None,
        *,
        source_version: str | None = None,
        expected_hpoa_sha256: str | None = None,
        expected_obo_sha256: str | None = None,
        ranking_strategy: RankingStrategy | str = RankingStrategy.IC_COVERAGE,
        disease_xref: str | Path | None = None,
    ) -> "HPOAgent2Matcher":
        """Create a matcher from local files without writing to either input.

        If ``hp_obo`` is omitted, a sibling file named ``hp.obo`` is used when
        present.  Exact matching remains available when no graph is found.
        ``disease_xref`` loads an OMIM/ORPHA cross-reference table (see
        ``load_disease_xref``) so that ``rank`` lists each disease once.
        """

        hpoa_path = Path(phenotype_hpoa)
        hpoa_sha256 = _sha256_file(hpoa_path)
        _verify_expected_sha256(hpoa_path, hpoa_sha256, expected_hpoa_sha256)
        profiles = _read_hpoa(hpoa_path)
        ontology: Optional[HPOOntology] = None
        obo_path: Optional[Path] = None
        if hp_obo is not None:
            obo_path = Path(hp_obo)
            ontology = HPOOntology.from_obo(obo_path)
        else:
            sibling = hpoa_path.with_name("hp.obo")
            if sibling.is_file():
                obo_path = sibling
                ontology = HPOOntology.from_obo(sibling)
        obo_sha256 = _sha256_file(obo_path) if obo_path is not None else None
        if obo_path is not None and obo_sha256 is not None:
            _verify_expected_sha256(obo_path, obo_sha256, expected_obo_sha256)
        elif expected_obo_sha256 is not None:
            raise ValueError("expected_obo_sha256 was provided but no hp.obo file was loaded")

        embedded_hpoa_version = _read_hpoa_version(hpoa_path)
        provenance = {
            "phenotype_hpoa": {
                "filename": hpoa_path.name,
                "sha256": hpoa_sha256,
                "version": source_version or embedded_hpoa_version,
                "embedded_version": embedded_hpoa_version,
            },
            "hp_obo": (
                {
                    "filename": obo_path.name,
                    "sha256": obo_sha256,
                    "version": ontology.version if ontology is not None else None,
                }
                if obo_path is not None
                else None
            ),
        }
        resolver = None
        if disease_xref is not None:
            resolver = load_disease_xref(disease_xref)
            provenance["disease_xref"] = {"source": resolver.source, "mapped_ids": len(resolver.mapping)}
        return cls(
            profiles,
            ontology,
            provenance=provenance,
            disease_resolver=resolver,
            ranking_strategy=ranking_strategy,
        )

    def rank(
        self,
        observations: Iterable[PhenotypeObservation | Mapping[str, Any] | Sequence[Any] | str],
        *,
        top_k: Optional[int] = 10,
        disease_ids: Optional[Iterable[str]] = None,
        ranking_strategy: RankingStrategy | str | None = None,
    ) -> list[CandidateScore]:
        """Return disease-profile rankings and auditable evidence.

        At least one known ``PRESENT`` or ``SUSPECTED`` phenotype is required.
        ``NOT_ASSESSED`` terms are retained in evidence with zero contribution.
        ``SUSPECTED`` terms contribute half the positive evidence of ``PRESENT``.
        Contradictory assessed states for the same HPO term raise ``ValueError``.
        ``ranking_strategy`` overrides the matcher's strategy for this call only.

        With a cross-reference resolver loaded, diseases sharing a concept are
        listed once, as their best-ranked member; the others are recorded in its
        ``provenance["equivalent_disease_ids"]`` and ``top_k`` counts concepts.
        An explicit ``disease_ids`` request is never merged.
        """

        strategy = self.ranking_strategy if ranking_strategy is None else RankingStrategy(ranking_strategy)
        normalized = self.sanitize_observations(observations)
        selected = self.profiles
        if disease_ids is not None:
            requested = {str(item) for item in disease_ids}
            missing = requested - set(self._profile_by_id)
            if missing:
                raise KeyError(f"Unknown disease profile(s): {sorted(missing)}")
            selected = tuple(self._profile_by_id[item] for item in sorted(requested))

        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative or None")
        if top_k == 0:
            return []

        scored: Iterable[CandidateScore] = (
            self._score_profile(profile, normalized) for profile in selected
        )
        if strategy is RankingStrategy.IC_COVERAGE:
            # Ties on coverage fall through to the likelihood score, then to the
            # same disease-ID order the likelihood strategy uses.
            ranking_key = lambda item: (
                item.ic_weighted_coverage,
                item.exact_coverage,
                item.score,
                item.disease_id,
            )
        else:
            ranking_key = lambda item: (item.score, item.disease_id)
        if self._calibrator is not None:
            scored = (self._with_confidence_score(item) for item in scored)
            if strategy is RankingStrategy.IC_COVERAGE:
                coverage_key = ranking_key
                ranking_key = lambda item: (item.confidence_score or 0.0, *coverage_key(item))
            else:
                ranking_key = lambda item: (item.confidence_score or 0.0, item.score)

        if disease_ids is None and not getattr(self.disease_resolver, "is_identity", False):
            return self._rank_merged(selected, scored, normalized, ranking_key, top_k)

        # Full HPOA contains thousands of disease profiles.  A bounded heap
        # keeps only the requested candidates instead of materialising every
        # evidence-rich CandidateScore in RAM.  top_k=None remains available
        # for small audits/tests that explicitly request the complete ranking.
        if top_k is None:
            return sorted(scored, key=ranking_key, reverse=True)
        return heapq.nlargest(top_k, scored, key=ranking_key)

    def _rank_merged(
        self,
        selected: Sequence[_DiseaseProfile],
        scored: Iterable[CandidateScore],
        observations: tuple[PhenotypeObservation, ...],
        ranking_key: Any,
        top_k: Optional[int],
    ) -> list[CandidateScore]:
        # Only each concept's best key and profile position are kept while
        # scanning; the few chosen profiles are rescored for their evidence.
        best: dict[str, tuple[Any, int]] = {}
        members: defaultdict[str, list[str]] = defaultdict(list)
        for position, item in enumerate(scored):
            concept = self.disease_resolver.resolve(item.disease_id)
            members[concept].append(item.disease_id)
            key = ranking_key(item)
            if concept not in best or key > best[concept][0]:
                best[concept] = (key, position)
        chosen = sorted(best.items(), key=lambda entry: entry[1][0], reverse=True)
        if top_k is not None:
            chosen = chosen[:top_k]
        results = []
        for concept, (_, position) in chosen:
            item = self._score_profile(selected[position], observations)
            if self._calibrator is not None:
                item = self._with_confidence_score(item)
            others = sorted(disease for disease in members[concept] if disease != item.disease_id)
            if others:
                item = replace(
                    item,
                    provenance={**item.provenance, "concept_id": concept, "equivalent_disease_ids": others},
                )
            results.append(item)
        return results

    # ------------------------------------------------------------------
    # probability-aware API
    # ------------------------------------------------------------------

    def analyze_agent1_request(
        self,
        payload: Mapping[str, Any] | str | bytes | bytearray,
        *,
        expected_payload_sha256: Optional[str] = None,
        priors: Optional[Any] = None,
        probability_mode: str = "binary",
        top_k: Optional[int] = 10,
        disease_ids: Optional[Iterable[str]] = None,
        require_probability: bool = False,
        require_calibrated: bool = False,
        unknown_prior: Optional[float] = None,
        min_candidate_coverage: float = 0.0,
        abstain_when_unknown_dominates: bool = False,
    ) -> Agent2Result:
        """Validate and analyze a direct ``profile_to_agent2_request`` payload.

        Contract failures never fall back to a loose observation parser.  They
        produce a structured ``INPUT_CONTRACT_REJECTED`` abstention with no
        candidates and therefore no ranking score or probability.  The trusted
        caller may bind exact JSON bytes with ``expected_payload_sha256``.
        """

        mode = str(probability_mode).strip().lower()
        if mode not in {"none", "binary", "single_cause"}:
            raise ValueError("probability_mode must be 'none', 'binary' or 'single_cause'")
        try:
            request = parse_agent1_to_agent2_request(
                payload, expected_payload_sha256=expected_payload_sha256
            )
        except Agent2ContractError as exc:
            error = exc.as_dict()
            return Agent2Result(
                candidates=(),
                observations=(),
                warnings=(),
                abstention=Abstention(
                    AbstentionReason.INPUT_CONTRACT_REJECTED,
                    "Agent-1 handoff rejected; no disease inference was performed.",
                    {"contract_error": error},
                ),
                context=None,
                probability_mode=mode,
                provenance={
                    **dict(self.provenance),
                    "agent1_contract": {"accepted": False, "error": error},
                },
            )

        result = self.analyze(
            request.observations,
            context=request.context,
            priors=priors,
            probability_mode=mode,
            top_k=top_k,
            disease_ids=disease_ids,
            require_probability=require_probability,
            require_calibrated=require_calibrated,
            unknown_prior=unknown_prior,
            min_candidate_coverage=min_candidate_coverage,
            abstain_when_unknown_dominates=abstain_when_unknown_dominates,
        )
        return replace(
            result,
            provenance={
                **dict(result.provenance),
                "agent1_contract": {
                    "accepted": True,
                    **request.provenance_dict(),
                    "upstream_warnings": list(request.warnings),
                },
            },
        )

    def analyze(
        self,
        observations: Iterable[Any],
        *,
        context: Optional[ClinicalContext] = None,
        priors: Optional[Any] = None,
        probability_mode: str = "binary",
        top_k: Optional[int] = 10,
        disease_ids: Optional[Iterable[str]] = None,
        require_probability: bool = False,
        require_calibrated: bool = False,
        unknown_prior: Optional[float] = None,
        min_candidate_coverage: float = 0.0,
        abstain_when_unknown_dominates: bool = False,
    ) -> Agent2Result:
        """Rank diseases and report probabilities only when they are earned.

        ``probability_mode``:

        ``"none"``
            Ranking only.  Every candidate reports ``RANKING_SCORE``.
        ``"binary"`` (default)
            Per-disease posteriors ``P(disease | evidence)``.  Requires a prior
            per disease; without one that disease falls back to
            ``RANKING_SCORE`` rather than inventing a number.  These do **not**
            sum to 1, because co-morbidity is possible.
        ``"single_cause"``
            Relative probabilities inside this candidate set under an explicit
            single-principal-cause assumption.  An ``OTHER_OR_UNKNOWN`` class is
            always added, so weak evidence yields "probably none of these"
            instead of a confident top-1.

        Rather than raising, unusable input produces a result whose
        ``abstention`` carries a machine-readable
        :class:`~hpo_agents.agent2_schema.AbstentionReason`.
        """

        mode = str(probability_mode).strip().lower()
        if mode not in {"none", "binary", "single_cause"}:
            raise ValueError("probability_mode must be 'none', 'binary' or 'single_cause'")

        warnings: list[Agent2Warning] = []
        provider = self._coerce_prior_provider(priors)

        release_abstention = self._hpo_release_abstention(context)
        if release_abstention is not None:
            return Agent2Result(
                candidates=(),
                observations=(),
                warnings=tuple(warnings),
                abstention=release_abstention,
                context=context,
                probability_mode=mode,
                provenance=dict(self.provenance),
            )

        try:
            detail = self._sanitation_detail(observations if observations is not None else ())
        except KeyError as exc:
            return self._abstain(
                AbstentionReason.UNKNOWN_HPO_ID,
                str(exc).strip("'\""),
                context=context,
                mode=mode,
                warnings=warnings,
            )
        except ValueError as exc:
            return self._abstain(
                AbstentionReason.UNRESOLVED_CONTRADICTION,
                str(exc),
                context=context,
                mode=mode,
                warnings=warnings,
            )

        warnings.extend(self._sanitation_warnings(detail))

        blocking = self._input_abstention(detail)
        if blocking is not None:
            return Agent2Result(
                candidates=(),
                observations=detail.observations,
                warnings=tuple(warnings),
                abstention=blocking,
                context=context,
                probability_mode=mode,
                provenance=dict(self.provenance),
            )

        candidates = self.rank(detail.observations, top_k=top_k, disease_ids=disease_ids)
        if not candidates:
            return self._abstain(
                AbstentionReason.LOW_CANDIDATE_COVERAGE,
                "No disease profile could be scored for these observations.",
                context=context,
                mode=mode,
                warnings=warnings,
                observations=detail.observations,
            )

        best_coverage = max(item.ancestor_coverage for item in candidates)
        if min_candidate_coverage > 0.0 and best_coverage < min_candidate_coverage:
            return self._abstain(
                AbstentionReason.LOW_CANDIDATE_COVERAGE,
                (
                    f"Best candidate covers {best_coverage:.2f} of the positive findings, "
                    f"below the required {min_candidate_coverage:.2f}."
                ),
                context=context,
                mode=mode,
                warnings=warnings,
                observations=detail.observations,
            )
        if best_coverage < 0.25:
            warnings.append(
                Agent2Warning(
                    WarningCode.LOW_CANDIDATE_COVERAGE,
                    "No candidate profile explains much of the observed phenotype set; "
                    "the ranking is weakly supported.",
                    {"best_ancestor_coverage": best_coverage},
                )
            )

        warnings.extend(self._model_assumption_warnings())

        likelihoods = {
            candidate.disease_id: self._likelihood_for(candidate) for candidate in candidates
        }
        for result in likelihoods.values():
            if result.capped_term_count:
                warnings.append(
                    Agent2Warning(
                        WarningCode.PER_TERM_CONTRIBUTION_CAPPED,
                        "At least one finding hit the per-term likelihood bound, so no single "
                        "annotation can dominate the estimate.",
                        {"max_abs_log_lr_per_term": self.likelihood_config.max_abs_log_lr_per_term},
                    )
                )
                break

        resolved_priors: dict[str, Optional[DiseasePrior]] = {
            candidate.disease_id: (
                provider.prior_for(candidate.disease_id, context) if provider else None
            )
            for candidate in candidates
        }
        missing_priors = [key for key, value in resolved_priors.items() if value is None]

        if mode != "none" and missing_priors:
            if require_probability and len(missing_priors) == len(candidates):
                return self._abstain(
                    AbstentionReason.PROBABILITY_REQUESTED_WITHOUT_PRIOR,
                    "A probability was requested but no disease prior was supplied for any "
                    "candidate; only a ranking can be produced.",
                    context=context,
                    mode=mode,
                    warnings=warnings,
                    observations=detail.observations,
                )
            warnings.append(
                Agent2Warning(
                    WarningCode.PRIOR_COVERAGE_INCOMPLETE
                    if len(missing_priors) < len(candidates)
                    else WarningCode.RANKING_ONLY_NO_PRIOR,
                    "No prior probability was available for "
                    f"{len(missing_priors)} of {len(candidates)} candidates; those are reported "
                    "as ranking scores without an absolute probability.",
                    {"diseases_without_prior": missing_priors[:20]},
                )
            )
        elif mode == "none":
            warnings.append(
                Agent2Warning(
                    WarningCode.RANKING_ONLY_NO_PRIOR,
                    "probability_mode='none': candidates are ordered but no absolute "
                    "probability is reported.",
                    {},
                )
            )

        calibrator_note = self._calibration_status()
        # ``require_calibrated`` is about the posterior, not about the ranking
        # calibrator, so it is gated on the stricter check.  Passing the ranking
        # check is necessary but not sufficient.
        posterior_calibration_note = self._posterior_calibration_status()
        if require_calibrated and posterior_calibration_note is not None:
            return self._abstain(
                AbstentionReason.CALIBRATOR_MISSING_OR_INCOMPATIBLE,
                posterior_calibration_note,
                context=context,
                mode=mode,
                warnings=warnings,
                observations=detail.observations,
            )

        unknown_payload: Optional[dict[str, Any]] = None
        posteriors: dict[str, PosteriorEstimate] = {}

        if mode == "single_cause":
            posteriors, unknown_payload, single_cause_warnings = self._single_cause_posteriors(
                candidates, likelihoods, resolved_priors, unknown_prior
            )
            warnings.extend(single_cause_warnings)
            if (
                abstain_when_unknown_dominates
                and unknown_payload
                and unknown_payload.get("unknown_dominates")
            ):
                return self._abstain(
                    AbstentionReason.UNKNOWN_CLASS_DOMINATES,
                    "The evidence fits none of the candidate diseases well; the "
                    f"{UNKNOWN_DISEASE_ID} class holds "
                    f"{unknown_payload['unknown_probability']:.2f} of the probability mass.",
                    context=context,
                    mode=mode,
                    warnings=warnings,
                    observations=detail.observations,
                )
        else:
            for candidate in candidates:
                log_lr = likelihoods[candidate.disease_id].log_likelihood_ratio
                prior = resolved_priors[candidate.disease_id]
                if mode == "none" or prior is None:
                    posteriors[candidate.disease_id] = _probability.ranking_only(log_lr)
                else:
                    posteriors[candidate.disease_id] = _probability.binary_posterior(
                        prior, log_lr, config=self.likelihood_config
                    )

        if any(
            estimate.kind == _probability.ProbabilityKind.BAYES_ESTIMATE_UNCALIBRATED
            for estimate in posteriors.values()
        ):
            warnings.append(
                Agent2Warning(
                    WarningCode.UNCALIBRATED_PROBABILITY,
                    "Probabilities are uncalibrated Bayes estimates. They have not been "
                    "checked against confirmed cases and must not be read as clinical "
                    "certainty.",
                    {"calibrator": calibrator_note or "not applied"},
                )
            )

        outputs: list[DiseaseCandidateOutput] = []
        for index, candidate in enumerate(candidates, start=1):
            matched, conflicting, ignored = self._partition_evidence(candidate.evidence)
            outputs.append(
                DiseaseCandidateOutput(
                    disease_id=candidate.disease_id,
                    disease_name=candidate.disease_name,
                    concept_id=self.disease_resolver.resolve(candidate.disease_id),
                    rank=index,
                    raw_score=candidate.score,
                    compatibility_score=candidate.compatibility_score,
                    posterior=posteriors[candidate.disease_id],
                    likelihood=likelihoods[candidate.disease_id],
                    matched_evidence=matched,
                    conflicting_evidence=conflicting,
                    ignored_evidence=ignored,
                    candidate=candidate,
                )
            )

        if getattr(self.disease_resolver, "is_identity", False):
            warnings.append(
                Agent2Warning(
                    WarningCode.DISEASE_IDENTITY_NOT_HARMONISED,
                    "OMIM/ORPHA/MONDO identifiers were not harmonised; one disease may appear "
                    "under more than one identifier. Load a curated cross-reference table to "
                    "merge them.",
                    {},
                )
            )
        warnings.extend(self._context_warnings(context))

        return Agent2Result(
            candidates=tuple(outputs),
            observations=detail.observations,
            warnings=tuple(_dedupe_warnings(warnings)),
            abstention=None,
            context=context,
            probability_mode=mode,
            unknown_class=unknown_payload,
            provenance={**dict(self.provenance), "ranking_strategy": self.ranking_strategy.value},
            calibrator_metadata=(
                self._calibrator_metadata.as_dict() if self._calibrator_metadata else None
            ),
        )

    # -- analyze() helpers ---------------------------------------------

    def _abstain(
        self,
        reason: AbstentionReason,
        message: str,
        *,
        context: Optional[ClinicalContext],
        mode: str,
        warnings: Sequence[Agent2Warning],
        observations: tuple[PhenotypeObservation, ...] = (),
        details: Optional[Mapping[str, Any]] = None,
    ) -> Agent2Result:
        return Agent2Result(
            candidates=(),
            observations=observations,
            warnings=tuple(_dedupe_warnings(list(warnings))),
            abstention=Abstention(reason, message, dict(details or {})),
            context=context,
            probability_mode=mode,
            provenance=dict(self.provenance),
        )

    def _hpo_release_abstention(
        self, context: Optional[ClinicalContext]
    ) -> Optional[Abstention]:
        if context is None or not context.hpo_release:
            return None
        loaded = (self.provenance.get("hp_obo") or {}).get("version") or self.ontology.version
        if loaded and str(loaded) != str(context.hpo_release):
            return Abstention(
                AbstentionReason.HPO_RELEASE_MISMATCH,
                f"Case was coded against HPO release {context.hpo_release!r} but the matcher "
                f"loaded {loaded!r}; re-code the case or load the matching snapshot.",
                {"case_release": context.hpo_release, "loaded_release": loaded},
            )
        return None

    def _sanitation_warnings(self, detail: "_SanitationResult") -> list[Agent2Warning]:
        warnings: list[Agent2Warning] = []
        if detail.ignored_root_terms:
            warnings.append(
                Agent2Warning(
                    WarningCode.ROOT_OBSERVATION_IGNORED,
                    f"{PHENOTYPIC_ABNORMALITY_ROOT} was supplied as an observation. It is true "
                    "of every abnormal phenotype, so it was accepted but excluded from scoring.",
                    {"terms": list(detail.ignored_root_terms)},
                )
            )
        if detail.merged_duplicate_terms:
            warnings.append(
                Agent2Warning(
                    WarningCode.DUPLICATE_MENTION_MERGED,
                    "Repeated mentions of the same HPO term were merged; their provenance is "
                    "kept but the likelihood counts each finding once.",
                    {"terms": list(detail.merged_duplicate_terms)},
                )
            )
        if detail.removed_redundant_terms:
            warnings.append(
                Agent2Warning(
                    WarningCode.REDUNDANT_ANCESTOR_REMOVED,
                    "Broader ancestor terms were dropped in favour of the more specific "
                    "observations so the same evidence is not counted twice.",
                    {"terms": list(detail.removed_redundant_terms)},
                )
            )
        return warnings

    def _input_abstention(self, detail: "_SanitationResult") -> Optional[Abstention]:
        if not detail.observations:
            if detail.ignored_root_terms:
                return Abstention(
                    AbstentionReason.ONLY_ROOT_PHENOTYPE,
                    f"The only phenotype supplied was {PHENOTYPIC_ABNORMALITY_ROOT}, which "
                    "cannot distinguish between diseases.",
                    {"terms": list(detail.ignored_root_terms)},
                )
            return Abstention(
                AbstentionReason.NO_VALID_HPO,
                "No usable HPO observation was supplied.",
                {},
            )
        assessed = [
            item
            for item in detail.observations
            if item.status != PhenotypeStatus.NOT_ASSESSED
        ]
        if not assessed:
            return Abstention(
                AbstentionReason.ALL_NOT_ASSESSED,
                "Every supplied phenotype is NOT_ASSESSED. Unassessed is not the same as "
                "absent, so nothing can be ranked.",
                {"terms": [item.hpo_id for item in detail.observations]},
            )
        positives = [
            item
            for item in assessed
            if item.status in {PhenotypeStatus.PRESENT, PhenotypeStatus.SUSPECTED}
        ]
        if not positives:
            return Abstention(
                AbstentionReason.NO_VALID_HPO,
                "Only absent findings were supplied; at least one PRESENT or SUSPECTED "
                "phenotype is required to rank diseases.",
                {"terms": [item.hpo_id for item in assessed]},
            )
        informative = [
            item
            for item in positives
            if self.information_content(item.hpo_id) >= _MIN_INFORMATIVE_IC
        ]
        if not informative:
            return Abstention(
                AbstentionReason.ONLY_VERY_GENERAL_TERMS,
                "Every positive finding is a very general term with almost no information "
                "content; a more specific phenotype is needed to rank diseases.",
                {
                    "terms": [item.hpo_id for item in positives],
                    "min_information_content": _MIN_INFORMATIVE_IC,
                },
            )
        return None

    def _model_assumption_warnings(self) -> list[Agent2Warning]:
        warnings = [
            Agent2Warning(
                WarningCode.NAIVE_CONDITIONAL_INDEPENDENCE,
                "Findings are combined as if they were conditionally independent given the "
                "disease. Related phenotypes violate that assumption; group capping limits "
                "but does not remove the resulting bias.",
                {
                    "max_abs_log_lr_per_group": self.likelihood_config.max_abs_log_lr_per_group,
                },
            )
        ]
        if self.background_model == BackgroundModel.ANNOTATION_BACKGROUND:
            warnings.append(
                Agent2Warning(
                    WarningCode.ANNOTATION_BACKGROUND_APPROXIMATION,
                    "P(HPO | not disease) is approximated by how often the term is annotated "
                    "across curated disease profiles. That is not its frequency in people "
                    "without the disease; replace it with a cohort background before "
                    "reporting probabilities.",
                    {"background_model": self.background_model.value},
                )
            )
        if self.missing_frequency_policy != MissingFrequencyPolicy.NEUTRAL:
            warnings.append(
                Agent2Warning(
                    WarningCode.MISSING_FREQUENCY_POLICY_APPLIED,
                    "Annotations without a curated frequency were handled by policy "
                    f"{self.missing_frequency_policy.value!r} rather than left neutral.",
                    {"policy": self.missing_frequency_policy.value},
                )
            )
        return warnings

    def _context_warnings(self, context: Optional[ClinicalContext]) -> list[Agent2Warning]:
        if context is None:
            return []
        retained: list[str] = []
        if context.gestational_age_weeks is not None:
            retained.append("gestational_age_weeks")
        if context.fetal_sex.value != "unknown":
            retained.append("fetal_sex")
        if context.usable_maternal_age is not None:
            retained.append("maternal_age_years")
        if context.family_history:
            retained.append("family_history")
        if not retained:
            return []
        return [
            Agent2Warning(
                WarningCode.CONTEXT_RETAINED_NOT_USED,
                "Clinical context was recorded but did not change any probability: no "
                "validated onset/sex/modifier likelihood is available yet. Missing context "
                "is treated as neutral and never penalised.",
                {"retained_fields": retained},
            )
        ]

    def _posterior_calibration_status(self) -> Optional[str]:
        """Return why no *calibrated posterior* can be produced, or ``None`` if one can.

        This is deliberately a different question from :meth:`_calibration_status`.
        The model attached by ``set_calibrator`` / ``fit_calibrator`` calibrates
        candidate **ranking confidence** (``CandidateScore.confidence_score``): it
        answers "is this candidate the right disease for this case", scored against
        the loaded disease universe.  That is not ``P(disease | evidence)`` and must
        never be relabelled as one.

        Agent 2 currently has no posterior calibrator, so a caller that demands a
        calibrated probability is refused even when a perfectly valid ranking
        calibrator is attached.  Returning the uncalibrated Bayes estimate instead
        would be the silent fallback this contract exists to prevent.
        """

        ranking_status = self._calibration_status()
        if ranking_status is not None:
            return ranking_status
        return (
            "the attached model calibrates candidate ranking confidence, not the "
            "disease posterior; Agent 2 has no posterior calibrator, so no "
            "CALIBRATED_PROBABILITY can be produced"
        )

    def _calibration_status(self) -> Optional[str]:
        """Return why the ranking calibrator may not be used, or ``None`` if it may."""

        if self._calibrator is None:
            return "no calibrator has been fitted or loaded"
        metadata = self._calibrator_metadata
        if metadata is None:
            return (
                "the fitted model has no CalibratorMetadata, so its cohort, split and HPO "
                "release cannot be verified"
            )
        loaded_release = (self.provenance.get("hp_obo") or {}).get("version") or self.ontology.version
        incompatible = metadata.incompatibility_with(
            feature_names=self.FEATURE_NAMES, hpo_release=loaded_release
        )
        if incompatible:
            return incompatible
        if not metadata.clinical_ready:
            return (
                "the calibrator is marked synthetic or has no clinician-reviewed held-out "
                "split, so it may be used for smoke tests only"
            )
        return None

    def _coerce_prior_provider(self, priors: Optional[Any]) -> Optional[DiseasePriorProvider]:
        if priors is None:
            return self.prior_provider
        if hasattr(priors, "prior_for"):
            return priors
        if isinstance(priors, Mapping):
            table: dict[str, DiseasePrior] = {}
            for key, value in priors.items():
                if not isinstance(value, DiseasePrior):
                    raise TypeError(
                        "Disease priors must be DiseasePrior objects so the population and "
                        f"source stay attached; got {type(value).__name__} for {key!r}"
                    )
                table[str(key)] = value
            from .agent2_schema import StaticDiseasePriorProvider

            return StaticDiseasePriorProvider(priors=table)
        raise TypeError(
            "priors must be a DiseasePriorProvider or a mapping of disease id -> DiseasePrior"
        )

    def _organ_system_group(self, hpo_id: str) -> Optional[str]:
        """Top-level organ system a term belongs to, used for dependency capping."""

        term = self.ontology.resolve(hpo_id)
        if term in self._group_cache:
            return self._group_cache[term]
        group: Optional[str] = None
        if PHENOTYPIC_ABNORMALITY_ROOT in self.ontology.known_terms:
            systems = sorted(
                node
                for node in self.ontology.ancestors_with_distance(term)
                if PHENOTYPIC_ABNORMALITY_ROOT in self.ontology.parents.get(node, frozenset())
            )
            group = systems[0] if systems else None
        self._group_cache[term] = group
        return group

    def _annotation_for(
        self, profile: _DiseaseProfile, term: Optional[str], *, excluded: bool
    ) -> Optional[HPOAAnnotation]:
        if not term:
            return None
        rows = profile.annotations.get(term) or []
        if not rows:
            return None
        if excluded:
            negatives = [row for row in rows if row.qualifier in {"NOT", "ABSENT", "EXCLUDED"}]
            if negatives:
                return negatives[0]
        positives = [row for row in rows if row.qualifier not in {"NOT", "ABSENT", "EXCLUDED"}]
        pool = positives or rows
        return max(pool, key=lambda row: row.frequency)

    def _frequency_estimate_for(
        self, profile: _DiseaseProfile, item: EvidenceItem
    ) -> _probability.FrequencyEstimate:
        background = float(item.background_probability or self._background_probability(item.observed_hpo_id))
        excluded = item.relation.startswith("profile_absent_")
        annotation = self._annotation_for(profile, item.matched_hpo_id, excluded=excluded)

        if excluded:
            return _probability.resolve_frequency(
                annotation.frequency_detail() if annotation else None,
                policy=self.missing_frequency_policy,
                background_probability=background,
                config=self.likelihood_config,
                excluded=True,
            )
        if item.relation == "no_match" or annotation is None:
            return _probability.FrequencyEstimate(
                effective_probability=_probability.clamp_probability(
                    self.likelihood_config.epsilon, epsilon=self.likelihood_config.epsilon
                ),
                contributes_to_likelihood=True,
                kind="absent_from_profile",
                policy_applied="not_annotated",
                note=(
                    "This curated profile does not list a comparable phenotype, so the "
                    "finding counts as evidence against it."
                ),
            )

        estimate = _probability.resolve_frequency(
            annotation.frequency_detail(),
            policy=self.missing_frequency_policy,
            background_probability=background,
            config=self.likelihood_config,
        )
        if estimate.kind == "missing" or item.disease_probability is None:
            # An unrecorded frequency must never become strong evidence: keep
            # whatever the configured policy decided.
            return estimate
        # For ancestor/semantic matches the matcher already discounted the
        # probability by match quality; keep that number but retain the curated
        # provenance next to it.
        return replace(
            estimate,
            effective_probability=_probability.clamp_probability(
                float(item.disease_probability), epsilon=self.likelihood_config.epsilon
            ),
            note=(
                f"{estimate.note} Effective P(HPO|disease) adjusted for a "
                f"{item.relation!r} match."
            ),
        )

    def _likelihood_for(self, candidate: CandidateScore) -> _probability.LogLikelihoodResult:
        """Rebuild a bounded, per-feature log-likelihood ratio for one candidate.

        This is deliberately *not* ``candidate.lirical_log_likelihood``: that
        value is the historical unbounded ranking term, whereas this one applies
        the per-term and per-group caps required before anything may be turned
        into a probability.
        """

        profile = self._profile_by_id[candidate.disease_id]
        terms: list[_probability.LikelihoodTerm] = []
        for item in candidate.evidence:
            if item.status == PhenotypeStatus.NOT_ASSESSED.value:
                continue
            observation = PhenotypeObservation(item.observed_hpo_id, item.status)
            weight = self.status_weight_policy.weight_for(observation)
            terms.append(
                _probability.build_likelihood_term(
                    hpo_id=item.observed_hpo_id,
                    status=item.status,
                    matched_hpo_id=item.matched_hpo_id,
                    relation=item.relation,
                    frequency=self._frequency_estimate_for(profile, item),
                    background_probability=float(
                        item.background_probability
                        or self._background_probability(item.observed_hpo_id)
                    ),
                    background_model=self.background_model,
                    weight=weight,
                    config=self.likelihood_config,
                    group_id=self._organ_system_group(item.observed_hpo_id),
                )
            )
        return _probability.combine_log_likelihood(terms, config=self.likelihood_config)

    def _single_cause_posteriors(
        self,
        candidates: Sequence[CandidateScore],
        likelihoods: Mapping[str, _probability.LogLikelihoodResult],
        resolved_priors: Mapping[str, Optional[DiseasePrior]],
        unknown_prior: Optional[float],
    ) -> tuple[dict[str, PosteriorEstimate], dict[str, Any], list[Agent2Warning]]:
        warnings: list[Agent2Warning] = []
        usable = [item for item in candidates if resolved_priors.get(item.disease_id) is not None]
        posteriors: dict[str, PosteriorEstimate] = {}

        if not usable:
            for candidate in candidates:
                posteriors[candidate.disease_id] = _probability.ranking_only(
                    likelihoods[candidate.disease_id].log_likelihood_ratio,
                    note=(
                        "single_cause mode needs a prior for at least one candidate; none was "
                        "supplied, so only a ranking is reported."
                    ),
                )
            return posteriors, {}, warnings

        entries = [
            SingleCauseEntry(
                disease_id=candidate.disease_id,
                log_likelihood_ratio=likelihoods[candidate.disease_id].log_likelihood_ratio,
                prior_probability=resolved_priors[candidate.disease_id].prior_probability,
                disease_name=candidate.disease_name,
            )
            for candidate in usable
        ]
        unknown_mass = (
            float(unknown_prior)
            if unknown_prior is not None
            else _probability.derive_unknown_prior(
                [entry.prior_probability for entry in entries]
            )
        )
        result = _probability.single_cause_posteriors(
            entries, UnknownClass(prior_probability=unknown_mass), config=self.likelihood_config
        )

        for candidate in candidates:
            log_lr = likelihoods[candidate.disease_id].log_likelihood_ratio
            prior = resolved_priors.get(candidate.disease_id)
            if prior is None:
                posteriors[candidate.disease_id] = _probability.ranking_only(
                    log_lr,
                    note=(
                        "No prior was supplied for this disease, so it was excluded from the "
                        "single-cause normalisation and is reported as a ranking score."
                    ),
                )
                continue
            posteriors[candidate.disease_id] = PosteriorEstimate(
                probability=float(result.probabilities[candidate.disease_id]),
                kind=_probability.ProbabilityKind.BAYES_ESTIMATE_UNCALIBRATED,
                scope=_probability.ProbabilityScope.SINGLE_CAUSE_CANDIDATE_SET,
                calibrated=False,
                prior=prior,
                log_prior_odds=None,
                log_likelihood_ratio=log_lr,
                log_posterior_odds=float(result.log_weights[candidate.disease_id]),
                notes=result.notes,
            )

        if result.unknown_dominates:
            warnings.append(
                Agent2Warning(
                    WarningCode.UNKNOWN_CLASS_LEADS,
                    f"The {UNKNOWN_DISEASE_ID} class holds "
                    f"{result.unknown_probability:.2f} of the probability mass: the findings "
                    "fit none of the candidate diseases well.",
                    {"unknown_probability": result.unknown_probability},
                )
            )
        payload = result.as_dict()
        payload["disease_id"] = UNKNOWN_DISEASE_ID
        payload["disease_name"] = UNKNOWN_DISEASE_NAME
        payload["prior_probability"] = unknown_mass
        return posteriors, payload, warnings

    @staticmethod
    def _partition_evidence(
        evidence: Sequence[EvidenceItem],
    ) -> tuple[tuple[EvidenceItem, ...], tuple[EvidenceItem, ...], tuple[EvidenceItem, ...]]:
        matched: list[EvidenceItem] = []
        conflicting: list[EvidenceItem] = []
        ignored: list[EvidenceItem] = []
        for item in evidence:
            if item.relation == "ignored" or item.status == PhenotypeStatus.NOT_ASSESSED.value:
                ignored.append(item)
            elif item.relation.startswith("profile_absent_") or item.weighted_contribution < 0.0:
                conflicting.append(item)
            elif item.relation == "no_match":
                ignored.append(item)
            else:
                matched.append(item)
        return tuple(matched), tuple(conflicting), tuple(ignored)

    def set_calibrator(
        self,
        model: Any,
        metadata: CalibratorMetadata,
    ) -> None:
        """Attach an externally fitted calibrator together with its provenance.

        The model is rejected outright when its feature vector or HPO release
        does not match this matcher, so a stale artefact fails loudly instead of
        producing quietly wrong probabilities.
        """

        if not hasattr(model, "predict_proba"):
            raise TypeError("A calibrator must expose predict_proba(X)")
        if not isinstance(metadata, CalibratorMetadata):
            raise TypeError("metadata must be a CalibratorMetadata instance")
        loaded_release = (self.provenance.get("hp_obo") or {}).get("version") or self.ontology.version
        incompatible = metadata.incompatibility_with(
            feature_names=self.FEATURE_NAMES, hpo_release=loaded_release
        )
        if incompatible:
            raise ValueError(f"Incompatible calibrator: {incompatible}")
        self._calibrator = model
        self._calibrator_metadata = metadata
        self.calibration_method = metadata.method

    @property
    def calibrator_metadata(self) -> Optional[CalibratorMetadata]:
        return self._calibrator_metadata

    def fit_calibrator(
        self,
        rows: Sequence[CandidateScore | Mapping[str, Any] | Sequence[float]],
        labels: Sequence[int | bool],
        *,
        method: str = "auto",
        random_state: int = 17,
        metadata: Optional[CalibratorMetadata] = None,
    ) -> str:
        """Fit an optional empirical ranking-confidence model.

        Feature order for numeric rows is ``FEATURE_NAMES``.  ``method='auto'``
        tries XGBoost, sklearn logistic regression, then NumPy logistic
        regression.  Explicit ``'xgboost'`` and ``'logistic'`` requests also
        fall back safely when their optional library is unavailable.
        """

        if len(rows) != len(labels) or not rows:
            raise ValueError("rows and labels must be non-empty and have equal length")
        target: list[int] = []
        for index, value in enumerate(labels):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Calibration label at index {index} is not numeric") from exc
            if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
                raise ValueError(f"Calibration label at index {index} must be binary 0 or 1")
            target.append(int(numeric))
        if len(set(target)) != 2:
            raise ValueError("calibration labels must contain both binary classes 0 and 1")
        matrix = self._coerce_feature_rows(rows)
        normalized_method = method.strip().lower()
        if normalized_method not in {"auto", "xgboost", "logistic", "sklearn", "numpy"}:
            raise ValueError("method must be auto, xgboost, logistic, sklearn, or numpy")

        if normalized_method in {"auto", "xgboost"}:
            try:
                from xgboost import XGBClassifier  # type: ignore

                model = XGBClassifier(
                    n_estimators=60,
                    max_depth=2,
                    learning_rate=0.05,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    eval_metric="logloss",
                    n_jobs=1,
                    random_state=random_state,
                    verbosity=0,
                )
                model.fit(matrix, target)
                return self._attach_fitted_calibrator(model, "xgboost", metadata)
            except (ImportError, ModuleNotFoundError, OSError, ValueError):
                pass

        if normalized_method in {"auto", "xgboost", "logistic", "sklearn"}:
            try:
                from sklearn.linear_model import LogisticRegression  # type: ignore

                model = LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                )
                model.fit(matrix, target)
                return self._attach_fitted_calibrator(model, "sklearn_logistic", metadata)
            except (ImportError, ModuleNotFoundError, OSError, ValueError):
                pass

        try:
            model = _NumpyLogisticCalibrator().fit(matrix, target)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Calibration requires xgboost, scikit-learn, or numpy; ranking itself has no such dependency"
            ) from exc
        return self._attach_fitted_calibrator(model, "numpy_logistic", metadata)

    def _attach_fitted_calibrator(
        self,
        model: Any,
        method: str,
        metadata: Optional[CalibratorMetadata],
    ) -> str:
        """Store a freshly fitted model, defaulting its provenance to synthetic.

        Without explicit metadata the result is recorded as synthetic and not
        clinician-reviewed, so ``analyze(require_calibrated=True)`` refuses it.
        Marking a calibrator clinically usable is a human decision backed by real
        confirmed cases, never a side effect of calling ``fit_calibrator``.
        """

        loaded_release = (self.provenance.get("hp_obo") or {}).get("version") or self.ontology.version
        if metadata is None:
            metadata = CalibratorMetadata(
                method=method,
                feature_names=self.FEATURE_NAMES,
                feature_version=FEATURE_VERSION,
                hpo_release=loaded_release,
                disease_universe=f"{len(self.profiles)} HPOA disease profiles",
                training_cohort="unspecified_in_process_fit",
                synthetic_only=True,
                clinician_reviewed=False,
                negative_sampling="not recorded by fit_calibrator; supply metadata to document it",
            )
        else:
            incompatible = metadata.incompatibility_with(
                feature_names=self.FEATURE_NAMES, hpo_release=loaded_release
            )
            if incompatible:
                raise ValueError(f"Incompatible calibrator metadata: {incompatible}")
        self._calibrator = model
        self._calibrator_metadata = metadata
        self.calibration_method = method
        return method

    def clear_calibrator(self) -> None:
        self._calibrator = None
        self.calibration_method = None
        self._calibrator_metadata = None

    def information_content(self, hpo_id: str) -> float:
        term = self.ontology.resolve(hpo_id)
        probability = self._background_probability(term)
        return -math.log(probability)

    def sanitize_observations(
        self,
        observations: Iterable[PhenotypeObservation | Mapping[str, Any] | Sequence[Any] | str],
    ) -> tuple[PhenotypeObservation, ...]:
        """Canonicalize and validate a patient phenotype profile.

        The public sanitation contract is shared by notebooks and ``rank``:
        obsolete/alternate IDs are resolved, unknown or non-phenotypic IDs and
        logical contradictions fail closed, and redundant ancestor observations
        are removed before scoring.
        """

        if observations is None:
            raise ValueError("At least one phenotype observation is required")
        detail = self._sanitation_detail(observations)
        normalized = detail.observations
        if not normalized:
            if detail.ignored_root_terms:
                raise ValueError(
                    f"Only the phenotype root {PHENOTYPIC_ABNORMALITY_ROOT} was supplied; "
                    "it applies to every abnormal phenotype and cannot rank diseases. "
                    "Use analyze() for a structured abstention instead of an exception."
                )
            raise ValueError("At least one phenotype observation is required")
        if not any(
            item.status in {PhenotypeStatus.PRESENT, PhenotypeStatus.SUSPECTED}
            for item in normalized
        ):
            raise ValueError(
                "At least one PRESENT or SUSPECTED phenotype is required; "
                "negative-only or unassessed profiles cannot be ranked safely"
            )
        return normalized

    def _canonicalize_profile(self, profile: _DiseaseProfile) -> _DiseaseProfile:
        positives: dict[str, float] = {}
        for term, frequency in profile.positive_frequencies.items():
            canonical = self.ontology.resolve(term)
            positives[canonical] = max(positives.get(canonical, 0.0), _clip_probability(frequency))
        negatives = {self.ontology.resolve(term) for term in profile.negative_terms}
        annotations: dict[str, list[HPOAAnnotation]] = defaultdict(list)
        for term, rows in profile.annotations.items():
            annotations[self.ontology.resolve(term)].extend(rows)
        for term in negatives:
            positives.pop(term, None)
        # The phenotype root is kept in ``annotations`` for audit but removed
        # from both scoring maps: it is true of every abnormal phenotype, so it
        # can neither support nor contradict any particular disease.  Profiles
        # keep all their other annotations, so no disease is lost.
        positives.pop(PHENOTYPIC_ABNORMALITY_ROOT, None)
        negatives.discard(PHENOTYPIC_ABNORMALITY_ROOT)
        return _DiseaseProfile(
            profile.disease_id,
            profile.disease_name,
            positives,
            negatives,
            dict(annotations),
        )

    def _build_document_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for profile in self.profiles:
            seen: set[str] = set()
            for term in profile.positive_frequencies:
                seen.update(self.ontology.ancestors_with_distance(term))
            for term in seen:
                counts[term] += 1
        return dict(counts)

    def _background_probability(self, term: str) -> float:
        resolved = self.ontology.resolve(term)
        if resolved == PHENOTYPIC_ABNORMALITY_ROOT:
            # Every annotated disease has *some* phenotypic abnormality, so the
            # root's background probability is ~1 and its information content
            # ~0.  Deriving it from annotation counts would instead make the
            # least specific term in the ontology look rare and informative.
            return 1.0 - _MIN_PHENOTYPE_PROBABILITY
        count = self._document_counts.get(resolved, 0)
        probability = (count + 0.5) / (len(self.profiles) + 1.0)
        return min(1.0 - _MIN_PHENOTYPE_PROBABILITY, max(_MIN_PHENOTYPE_PROBABILITY, probability))

    def _normalize_observations(self, values: Iterable[Any]) -> tuple[PhenotypeObservation, ...]:
        return self._sanitation_detail(values).observations

    def _sanitation_detail(self, values: Iterable[Any]) -> "_SanitationResult":
        """Canonicalise observations and report everything that was changed.

        Beyond the tuple of scorable observations this records what was dropped
        and why, so ``analyze`` can turn it into machine-readable warnings
        instead of silently discarding a clinician's input.
        """

        ordered_terms: list[str] = []
        merged: dict[str, PhenotypeObservation] = {}
        ignored_root: list[str] = []
        duplicate_terms: list[str] = []
        unknown_terms: list[str] = []

        for raw in values:
            observation = PhenotypeObservation.coerce(raw)
            term = self.ontology.resolve(observation.hpo_id)

            # The phenotype root is a legal term, so it is accepted rather than
            # raising; it just cannot discriminate, so it never reaches scoring.
            if self.ontology.is_phenotype_root(term):
                ignored_root.append(term)
                continue

            if self.ontology.known_terms:
                is_known = self.ontology.contains(term)
            else:
                is_known = term in self._known_profile_terms
            if not is_known:
                unknown_terms.append(observation.hpo_id)
                raise KeyError(f"Unknown HPO term in loaded snapshot: {observation.hpo_id}")
            if not self.ontology.is_within_phenotypic_abnormality(term):
                raise ValueError(f"HPO term is not a descendant of HP:0000118: {term}")

            canonical = replace(observation, hpo_id=term)
            status = canonical.status
            if term not in merged:
                ordered_terms.append(term)
                merged[term] = canonical
                continue

            # Repeated mentions of one finding merge their provenance but are
            # scored once; multiplying the likelihood per mention would let a
            # verbose report outweigh a concise one.
            previous = merged[term]
            duplicate_terms.append(term)
            if previous.status == status:
                combined = status
            elif previous.status == PhenotypeStatus.NOT_ASSESSED:
                combined = status
            elif status == PhenotypeStatus.NOT_ASSESSED:
                combined = previous.status
            elif {previous.status, status} == {PhenotypeStatus.PRESENT, PhenotypeStatus.SUSPECTED}:
                combined = PhenotypeStatus.PRESENT
            else:
                raise ValueError(
                    f"Contradictory assessed states for {term}: "
                    f"{previous.status.value} and {status.value}"
                )
            merged[term] = previous.merged_with(canonical, combined)

        states = {term: merged[term].status for term in ordered_terms}
        positive_states = {PhenotypeStatus.PRESENT, PhenotypeStatus.SUSPECTED}
        positive_terms = [term for term in ordered_terms if states[term] in positive_states]
        absent_terms = [term for term in ordered_terms if states[term] == PhenotypeStatus.ABSENT]

        for absent_term in absent_terms:
            for positive_term in positive_terms:
                if self.ontology.is_ancestor(absent_term, positive_term):
                    raise ValueError(
                        "Contradictory ancestor/descendant states: "
                        f"{absent_term} is ABSENT but descendant {positive_term} is "
                        f"{states[positive_term].value}"
                    )

        strength = {PhenotypeStatus.SUSPECTED: 1, PhenotypeStatus.PRESENT: 2}
        redundant: set[str] = set()
        for broader in positive_terms:
            for narrower in positive_terms:
                if broader == narrower:
                    continue
                if (
                    self.ontology.is_ancestor(broader, narrower)
                    and strength[states[narrower]] >= strength[states[broader]]
                ):
                    redundant.add(broader)
                    break
        # If a broad phenotype is absent, its absent descendants add no new
        # information and would otherwise double-count negative evidence.
        for narrower in absent_terms:
            if any(
                broader != narrower and self.ontology.is_ancestor(broader, narrower)
                for broader in absent_terms
            ):
                redundant.add(narrower)

        return _SanitationResult(
            observations=tuple(
                merged[term] for term in ordered_terms if term not in redundant
            ),
            ignored_root_terms=tuple(dict.fromkeys(ignored_root)),
            merged_duplicate_terms=tuple(dict.fromkeys(duplicate_terms)),
            removed_redundant_terms=tuple(sorted(redundant)),
            unknown_terms=tuple(dict.fromkeys(unknown_terms)),
        )

    def _score_profile(
        self,
        profile: _DiseaseProfile,
        observations: tuple[PhenotypeObservation, ...],
    ) -> CandidateScore:
        positive_weight_total = 0.0
        exact_weight = 0.0
        ancestor_weight = 0.0
        ic_denominator = 0.0
        ic_numerator = 0.0
        absent_ic_denominator = 0.0
        negative_conflict_numerator = 0.0
        log_likelihood = 0.0
        evidence: list[EvidenceItem] = []

        for observation in observations:
            if observation.status == PhenotypeStatus.NOT_ASSESSED:
                evidence.append(
                    EvidenceItem(
                        observed_hpo_id=observation.hpo_id,
                        status=observation.status.value,
                        matched_hpo_id=None,
                        relation="ignored",
                        distance=None,
                        ic_similarity=0.0,
                        disease_frequency=None,
                        disease_probability=None,
                        background_probability=None,
                        log_likelihood_ratio=0.0,
                        weighted_contribution=0.0,
                        message="Not assessed; excluded from coverage, conflict and likelihood calculations.",
                    )
                )
                continue

            positive_match = self._best_match(observation.hpo_id, profile.positive_frequencies)
            negative_match = self._best_match(
                observation.hpo_id,
                {term: 0.0 for term in profile.negative_terms},
            )
            # A curated NOT annotation only applies to the same term or a
            # directly comparable ancestor/descendant.  Shared-ancestor
            # similarity between siblings is too weak to infer explicit
            # absence and would create unsafe false conflicts.
            if negative_match is not None and negative_match.relation == "semantic":
                negative_match = None
            disease_probability, chosen_match, explicit_profile_absence = self._disease_probability(
                positive_match,
                negative_match,
            )
            background_probability = self._background_probability(observation.hpo_id)
            # Evidence weight now comes from the configured policy: it reduces to
            # the historical 1.0 / 0.5 when Agent 1 supplies no confidence, and
            # is derived from extraction/linking confidence and reviewer status
            # when it does.
            status_weight = self.status_weight_policy.weight_for(observation)

            if observation.status in {PhenotypeStatus.PRESENT, PhenotypeStatus.SUSPECTED}:
                raw_llr = math.log(disease_probability / background_probability)
                contribution = status_weight * raw_llr
                positive_weight_total += status_weight
                information = self.information_content(observation.hpo_id)
                semantic_similarity = (
                    positive_match.ic_similarity if positive_match is not None and not explicit_profile_absence else 0.0
                )
                if positive_match is not None and not explicit_profile_absence:
                    if positive_match.relation == "exact":
                        exact_weight += status_weight
                    if positive_match.relation in {"exact", "profile_ancestor", "profile_descendant"}:
                        ancestor_weight += status_weight
                ic_denominator += status_weight * information
                ic_numerator += status_weight * information * semantic_similarity
            else:
                raw_llr = math.log(
                    (1.0 - disease_probability) / (1.0 - background_probability)
                )
                # Confirmed absences are also weighted: a low-confidence
                # negative should not veto a disease as hard as a clinician's.
                contribution = status_weight * raw_llr
                information = self.information_content(observation.hpo_id)
                absent_ic_denominator += information
                if positive_match is not None:
                    negative_conflict_numerator += information * disease_probability

            log_likelihood += contribution
            evidence.append(
                self._make_evidence(
                    profile,
                    observation,
                    positive_match,
                    chosen_match,
                    explicit_profile_absence,
                    disease_probability,
                    background_probability,
                    raw_llr,
                    contribution,
                )
            )

        exact_coverage = exact_weight / positive_weight_total if positive_weight_total else 0.0
        ancestor_coverage = ancestor_weight / positive_weight_total if positive_weight_total else 0.0
        if ic_denominator > _EPSILON:
            ic_weighted_coverage = ic_numerator / ic_denominator
        elif positive_weight_total:
            ic_weighted_coverage = ancestor_weight / positive_weight_total
        else:
            ic_weighted_coverage = 0.0
        negative_conflict = (
            negative_conflict_numerator / absent_ic_denominator
            if absent_ic_denominator > _EPSILON
            else 0.0
        )
        negative_conflict = min(1.0, max(0.0, negative_conflict))

        # The log likelihood remains the dominant term.  Small bounded bonuses
        # make transparent match quality useful for deterministic tie-breaking.
        score = (
            log_likelihood
            + 0.50 * ic_weighted_coverage
            + 0.25 * exact_coverage
            + 0.10 * ancestor_coverage
            - 0.75 * negative_conflict
        )
        compatibility_score = min(
            1.0,
            max(
                0.0,
                0.45 * ic_weighted_coverage
                + 0.30 * exact_coverage
                + 0.25 * ancestor_coverage
                - 0.50 * negative_conflict,
            ),
        )
        explanations = (
            f"Exact positive-term coverage: {exact_coverage:.3f}.",
            f"Exact-or-ancestor positive-term coverage: {ancestor_coverage:.3f}.",
            f"Information-content-weighted semantic coverage: {ic_weighted_coverage:.3f}.",
            f"Conflict from explicitly absent observations: {negative_conflict:.3f}.",
            f"LIRICAL-like log-likelihood ratio: {log_likelihood:.3f}.",
            "The score ranks curated profiles and is not a clinical conclusion.",
        )
        return CandidateScore(
            disease_id=profile.disease_id,
            disease_name=profile.disease_name,
            score=float(score),
            compatibility_score=float(compatibility_score),
            confidence_score=None,
            exact_coverage=float(exact_coverage),
            ancestor_coverage=float(ancestor_coverage),
            ic_weighted_coverage=float(ic_weighted_coverage),
            negative_conflict=float(negative_conflict),
            lirical_log_likelihood=float(log_likelihood),
            calibration_method=None,
            evidence=tuple(evidence),
            explanation=explanations,
            provenance=self.provenance,
        )

    def _make_evidence(
        self,
        profile: _DiseaseProfile,
        observation: PhenotypeObservation,
        positive_match: Optional[_Match],
        chosen_match: Optional[_Match],
        explicit_profile_absence: bool,
        disease_probability: float,
        background_probability: float,
        raw_llr: float,
        contribution: float,
    ) -> EvidenceItem:
        match = chosen_match
        if explicit_profile_absence:
            relation = f"profile_absent_{match.relation if match else 'match'}"
            message = "The curated disease profile explicitly marks a comparable phenotype as absent."
        elif positive_match is None:
            relation = "no_match"
            message = "No comparable positive phenotype annotation was found in this disease profile."
        else:
            relation = positive_match.relation
            message = {
                "exact": "Exact HPO term match in the curated disease profile.",
                "profile_ancestor": "The disease profile contains a broader ancestor term.",
                "profile_descendant": "The disease profile contains a more specific descendant term.",
                "semantic": "The terms share an informative ancestor but are not directly comparable.",
            }.get(relation, "Comparable phenotype evidence was found.")
        return EvidenceItem(
            observed_hpo_id=observation.hpo_id,
            status=observation.status.value,
            matched_hpo_id=match.term if match else None,
            relation=relation,
            distance=match.distance if match else None,
            ic_similarity=float(match.ic_similarity if match else 0.0),
            disease_frequency=(None if explicit_profile_absence else (match.frequency if match else None)),
            disease_probability=float(disease_probability),
            background_probability=float(background_probability),
            log_likelihood_ratio=float(raw_llr),
            weighted_contribution=float(contribution),
            message=message,
            hpoa_annotations=tuple(profile.annotations.get(match.term, ())) if match else (),
        )

    def _disease_probability(
        self,
        positive_match: Optional[_Match],
        negative_match: Optional[_Match],
    ) -> tuple[float, Optional[_Match], bool]:
        if negative_match is not None and (
            positive_match is None or self._match_priority(negative_match) >= self._match_priority(positive_match)
        ):
            return _MIN_PHENOTYPE_PROBABILITY, negative_match, True
        if positive_match is None:
            return _MIN_PHENOTYPE_PROBABILITY, None, False
        if positive_match.relation in {"exact", "profile_descendant"}:
            probability = positive_match.frequency
        elif positive_match.relation == "profile_ancestor":
            probability = _MIN_PHENOTYPE_PROBABILITY + (
                positive_match.frequency - _MIN_PHENOTYPE_PROBABILITY
            ) * positive_match.strength
        else:
            probability = _MIN_PHENOTYPE_PROBABILITY + (
                positive_match.frequency - _MIN_PHENOTYPE_PROBABILITY
            ) * 0.25 * positive_match.strength
        return _clip_probability(probability), positive_match, False

    def _best_match(self, query: str, frequencies: Mapping[str, float]) -> Optional[_Match]:
        best: Optional[_Match] = None
        for term, frequency in frequencies.items():
            candidate = self._compare_terms(query, term, frequency)
            if candidate is None:
                continue
            if best is None or self._match_priority(candidate) > self._match_priority(best):
                best = candidate
        return best

    @staticmethod
    def _match_priority(match: _Match) -> tuple[float, float, float, str]:
        relation_priority = {
            "exact": 4.0,
            "profile_descendant": 3.0,
            "profile_ancestor": 3.0,
            "semantic": 1.0,
        }.get(match.relation, 0.0)
        distance = float(match.distance if match.distance is not None else 1_000_000)
        return (relation_priority, match.strength * max(match.frequency, 0.01), -distance, match.term)

    def _compare_terms(self, query: str, profile_term: str, frequency: float) -> Optional[_Match]:
        query = self.ontology.resolve(query)
        profile_term = self.ontology.resolve(profile_term)
        if query == profile_term:
            return _Match(profile_term, "exact", 0, 1.0, 1.0, _clip_probability(frequency))

        query_ancestors = self.ontology.ancestors_with_distance(query)
        profile_ancestors = self.ontology.ancestors_with_distance(profile_term)
        query_information = self.information_content(query)
        if query in profile_ancestors:
            distance = profile_ancestors[query]
            return _Match(
                profile_term,
                "profile_descendant",
                distance,
                1.0,
                1.0,
                _clip_probability(frequency),
            )
        if profile_term in query_ancestors:
            distance = query_ancestors[profile_term]
            ancestor_information = self.information_content(profile_term)
            similarity = (
                min(1.0, ancestor_information / query_information)
                if query_information > _EPSILON
                else 1.0 / (1.0 + distance)
            )
            strength = max(similarity, 0.5**distance)
            return _Match(
                profile_term,
                "profile_ancestor",
                distance,
                similarity,
                strength,
                _clip_probability(frequency),
            )

        common = set(query_ancestors).intersection(profile_ancestors)
        if not common:
            return None
        mica = max(common, key=lambda term: (self.information_content(term), term))
        mica_information = self.information_content(mica)
        if mica_information <= _EPSILON:
            return None
        similarity = min(1.0, mica_information / max(query_information, _EPSILON))
        distance = query_ancestors[mica] + profile_ancestors[mica]
        return _Match(
            profile_term,
            "semantic",
            distance,
            similarity,
            similarity,
            _clip_probability(frequency),
        )

    def _with_confidence_score(self, item: CandidateScore) -> CandidateScore:
        confidence_score = float(self._calibrator.predict_proba([item.feature_vector()])[0][1])
        if not math.isfinite(confidence_score):
            raise ValueError("Confidence model returned a non-finite score")
        return CandidateScore(
            disease_id=item.disease_id,
            disease_name=item.disease_name,
            score=item.score,
            compatibility_score=item.compatibility_score,
            confidence_score=min(1.0, max(0.0, confidence_score)),
            exact_coverage=item.exact_coverage,
            ancestor_coverage=item.ancestor_coverage,
            ic_weighted_coverage=item.ic_weighted_coverage,
            negative_conflict=item.negative_conflict,
            lirical_log_likelihood=item.lirical_log_likelihood,
            calibration_method=self.calibration_method,
            evidence=item.evidence,
            explanation=item.explanation,
            provenance=item.provenance,
            scope_note=item.scope_note,
        )

    def _coerce_feature_rows(self, rows: Sequence[Any]) -> list[list[float]]:
        matrix: list[list[float]] = []
        for row in rows:
            if isinstance(row, CandidateScore):
                values = row.feature_vector()
            elif isinstance(row, Mapping):
                metrics = row.get("metrics", row)
                values = tuple(float(metrics[name]) for name in self.FEATURE_NAMES)
            else:
                values = tuple(float(value) for value in row)
                if len(values) != len(self.FEATURE_NAMES):
                    raise ValueError(
                        f"Numeric calibration rows require {len(self.FEATURE_NAMES)} features"
                    )
            numeric_values = [float(value) for value in values]
            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError("Calibration features must all be finite")
            matrix.append(numeric_values)
        return matrix


def _normalize_hpo_id(value: Any) -> str:
    term = str(value).strip().upper().replace("_", ":")
    if term.startswith("HP") and ":" not in term and term[2:].isdigit():
        term = f"HP:{term[2:]}"
    if not _HPO_ID_RE.fullmatch(term):
        raise ValueError(f"Invalid HPO identifier: {value!r}")
    return term


def _clip_probability(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Phenotype frequency must be finite")
    return min(1.0 - _MIN_PHENOTYPE_PROBABILITY, max(_MIN_PHENOTYPE_PROBABILITY, numeric))


def _normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().casefold())


def _parse_frequency(value: str) -> float:
    """Representative frequency only.

    ``_read_hpoa`` uses :func:`~hpo_agents.ontology.parse_frequency_detail`
    instead, because the probability layer needs to know whether a number was
    curated or defaulted.  This thin wrapper is kept for callers that only want
    the scalar.
    """

    return parse_frequency(value, default=_DEFAULT_DISEASE_FREQUENCY)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_expected_sha256(path: Path, actual: str, expected: str | None) -> None:
    if expected is None:
        return
    normalized = expected.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"Expected SHA-256 for {path.name} must contain 64 hexadecimal characters")
    if actual != normalized:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {normalized}, observed {actual}"
        )


def _read_hpoa_version(path: Path) -> str | None:
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_number, raw_line in enumerate(stream):
            if line_number >= 100:
                break
            stripped = raw_line.strip()
            if not stripped.startswith("#"):
                break
            match = re.match(r"#\s*(?:version|release|date)\s*[:=]\s*(.+)$", stripped, re.IGNORECASE)
            if match:
                return match.group(1).strip() or None
    return None


def _read_hpoa(path: Path) -> list[_DiseaseProfile]:
    """Read modern or legacy tab-separated ``phenotype.hpoa`` files."""

    standard_columns = [
        "database_id",
        "disease_name",
        "qualifier",
        "hpo_id",
        "reference",
        "evidence",
        "onset",
        "frequency",
        "sex",
        "modifier",
        "aspect",
        "biocuration",
    ]
    aliases = {
        "database_id": {"databaseid", "diseaseid", "database_id"},
        "disease_name": {"diseasename", "disease_name"},
        "qualifier": {"qualifier"},
        "hpo_id": {"hpoid", "hpo_id"},
        "frequency": {"frequency"},
        "aspect": {"aspect"},
        "reference": {"reference", "db_reference", "dbreference"},
        "evidence": {"evidence"},
        "onset": {"onset"},
        "sex": {"sex"},
        "modifier": {"modifier"},
        "biocuration": {"biocuration"},
    }
    header: Optional[list[str]] = None
    records: dict[str, _DiseaseProfile] = {}

    def find_index(columns: Sequence[str], field: str) -> Optional[int]:
        normalized = [_normalize_column(item) for item in columns]
        expected = {_normalize_column(item) for item in aliases[field]}
        for index, name in enumerate(normalized):
            if name in expected:
                return index
        return None

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        for raw_line in stream:
            stripped = raw_line.rstrip("\r\n")
            if not stripped.strip():
                continue
            candidate = stripped[1:] if stripped.startswith("#") else stripped
            fields = candidate.split("\t")
            candidate_norm = {_normalize_column(item) for item in fields}
            looks_like_header = "hpoid" in candidate_norm and bool(
                {"databaseid", "diseaseid"}.intersection(candidate_norm)
            )
            if looks_like_header:
                header = fields
                continue
            if stripped.startswith("#"):
                continue
            columns = header or standard_columns
            hpo_index = find_index(columns, "hpo_id")
            disease_index = find_index(columns, "database_id")
            name_index = find_index(columns, "disease_name")
            qualifier_index = find_index(columns, "qualifier")
            frequency_index = find_index(columns, "frequency")
            aspect_index = find_index(columns, "aspect")
            reference_index = find_index(columns, "reference")
            evidence_index = find_index(columns, "evidence")
            onset_index = find_index(columns, "onset")
            sex_index = find_index(columns, "sex")
            modifier_index = find_index(columns, "modifier")
            biocuration_index = find_index(columns, "biocuration")
            if hpo_index is None or disease_index is None:
                raise ValueError("phenotype.hpoa header must contain disease/database ID and HPO ID")

            def get(index: Optional[int], default: str = "") -> str:
                return fields[index].strip() if index is not None and index < len(fields) else default

            disease_id = get(disease_index)
            hpo_id_raw = get(hpo_index)
            aspect = get(aspect_index).upper()
            if aspect and aspect != "P":
                continue
            if not disease_id or not hpo_id_raw:
                continue
            hpo_id = _normalize_hpo_id(hpo_id_raw)
            disease_name = get(name_index, disease_id) or disease_id
            qualifier = get(qualifier_index).upper()
            frequency_detail = parse_frequency_detail(
                get(frequency_index), default=_DEFAULT_DISEASE_FREQUENCY
            )
            frequency = float(frequency_detail["probability"])
            profile = records.setdefault(
                disease_id,
                _DiseaseProfile(disease_id, disease_name, {}, set()),
            )
            profile.annotations.setdefault(hpo_id, []).append(
                HPOAAnnotation(
                    hpo_id=hpo_id,
                    qualifier=qualifier,
                    frequency=float(frequency),
                    reference=get(reference_index) or None,
                    evidence=get(evidence_index) or None,
                    onset=get(onset_index) or None,
                    sex=get(sex_index) or None,
                    modifier=get(modifier_index) or None,
                    aspect=aspect or None,
                    biocuration=get(biocuration_index) or None,
                    frequency_raw=frequency_detail["raw"],
                    frequency_kind=str(frequency_detail["kind"]),
                    frequency_numerator=frequency_detail["numerator"],
                    frequency_denominator=frequency_detail["denominator"],
                    frequency_category_term=frequency_detail["category_term"],
                    frequency_category_label=frequency_detail["category_label"],
                )
            )
            is_negative = qualifier in {"NOT", "ABSENT", "EXCLUDED"} or frequency <= 0.0
            if is_negative:
                profile.negative_terms.add(hpo_id)
                profile.positive_frequencies.pop(hpo_id, None)
            elif hpo_id not in profile.negative_terms:
                profile.positive_frequencies[hpo_id] = max(
                    profile.positive_frequencies.get(hpo_id, 0.0),
                    _clip_probability(frequency),
                )
    if not records:
        raise ValueError(f"No disease annotations found in {path}")
    return [records[key] for key in sorted(records)]


__all__ = [
    "AGENT2_SCHEMA_VERSION",
    "FEATURE_VERSION",
    "PHENOTYPIC_ABNORMALITY_ROOT",
    "UNKNOWN_DISEASE_ID",
    "Abstention",
    "AbstentionReason",
    "Agent2Result",
    "Agent2Warning",
    "BackgroundModel",
    "CalibratorMetadata",
    "CandidateScore",
    "ClinicalContext",
    "DiseaseCandidateOutput",
    "DiseasePrior",
    "EvidenceItem",
    "HPOAAnnotation",
    "HPOAgent2Matcher",
    "HPOOntology",
    "LikelihoodConfig",
    "MissingFrequencyPolicy",
    "PhenotypeObservation",
    "PhenotypeStatus",
    "PosteriorEstimate",
    "ReviewerStatus",
    "StatusWeightPolicy",
    "WarningCode",
]
