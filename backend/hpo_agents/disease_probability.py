"""Disease-probability layer for Agent 2.

This module is pure arithmetic with no I/O and no ontology dependency, so it can
be unit-tested on its own.  It exists to keep three things apart that the rest
of the pipeline must never confuse:

``RANKING_SCORE``
    A number that orders candidates.  Meaningless on an absolute scale.  This is
    what you get when no disease prior is available.

``BAYES_ESTIMATE_UNCALIBRATED``
    ``posterior_odds = prior_odds * prod(LR_i)``, evaluated in log space.  It is
    a real probability *under the stated model assumptions* (a supplied prior, a
    background model, and naive conditional independence between findings), but
    it has never been checked against confirmed cases.

``CALIBRATED_PROBABILITY``
    The above, mapped through a calibrator that was fitted and validated on
    clinician-confirmed cases with a patient-disjoint held-out split.

Likelihood ratios
-----------------
For an observed finding::

    LR_present = P(HPO | disease) / P(HPO | not disease)

For a finding a clinician explicitly confirmed to be absent::

    LR_absent = (1 - P(HPO | disease)) / (1 - P(HPO | not disease))

Everything is accumulated as ``log LR`` so that long evidence lists cannot
underflow or overflow, every term is clipped away from 0 and 1 so sparse data
cannot produce an infinite ratio, and each term's contribution is bounded so a
single mis-curated annotation cannot dominate a case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .agent2_schema import (
    BackgroundModel,
    DiseasePrior,
    MissingFrequencyPolicy,
    ProbabilityKind,
    ProbabilityScope,
)


#: Reserved identifier for the "not one of the listed diseases" class.
UNKNOWN_DISEASE_ID = "OTHER_OR_UNKNOWN"
UNKNOWN_DISEASE_NAME = "Other or not represented in the knowledge base"


@dataclass(frozen=True)
class LikelihoodConfig:
    """Numeric guard-rails for the likelihood and posterior computation."""

    #: Phenotype probabilities are clipped into ``[epsilon, 1 - epsilon]``.
    epsilon: float = 1e-4
    #: Reported probabilities are clipped into ``[floor, 1 - floor]`` so a
    #: sparse knowledge base can never print "0%" or "100%".
    reported_probability_floor: float = 1e-6
    #: Hard bound on ``|log LR|`` for a single finding.  ``5.0`` is a likelihood
    #: ratio of about 148:1 in either direction.
    max_abs_log_lr_per_term: float = 5.0
    #: Optional bound on the summed ``|log LR|`` of all findings that share one
    #: organ-system ancestor.  ``None`` disables group capping.
    max_abs_log_lr_per_group: Optional[float] = 8.0
    #: Pseudo-counts used when a frequency arrives as an ``n/m`` ratio.
    beta_pseudo_count: float = 0.5
    #: Explicit default used only by ``MissingFrequencyPolicy.CONFIGURABLE_DEFAULT``.
    missing_frequency_default: float = 0.5
    #: A single-cause differential whose unknown class exceeds this share is
    #: reported as "unknown dominates".
    unknown_dominance_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon < 0.5:
            raise ValueError("epsilon must satisfy 0 < epsilon < 0.5")
        if not 0.0 < self.reported_probability_floor < 0.5:
            raise ValueError("reported_probability_floor must satisfy 0 < floor < 0.5")
        if not self.max_abs_log_lr_per_term > 0.0:
            raise ValueError("max_abs_log_lr_per_term must be positive")
        if self.max_abs_log_lr_per_group is not None and self.max_abs_log_lr_per_group <= 0.0:
            raise ValueError("max_abs_log_lr_per_group must be positive or None")
        if self.beta_pseudo_count < 0.0:
            raise ValueError("beta_pseudo_count must be non-negative")


DEFAULT_CONFIG = LikelihoodConfig()


# --------------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------------


def clamp_probability(value: float, *, epsilon: float = DEFAULT_CONFIG.epsilon) -> float:
    """Clip a probability into ``[epsilon, 1 - epsilon]``, rejecting NaN/inf."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"probability must be finite; got {value!r}")
    return min(1.0 - epsilon, max(epsilon, number))


def log_odds(probability: float) -> float:
    """``log(p / (1 - p))`` computed without cancelling significant digits."""

    p = clamp_probability(probability)
    return math.log(p) - math.log1p(-p)


def probability_from_log_odds(value: float) -> float:
    """Inverse of :func:`log_odds`, overflow-safe in both tails."""

    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"log-odds must be finite; got {value!r}")
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exponential = math.exp(x)
    return exponential / (1.0 + exponential)


def log_sum_exp(values: Sequence[float]) -> float:
    """Numerically stable ``log(sum(exp(v)))``."""

    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        raise ValueError("log_sum_exp requires at least one finite value")
    largest = max(finite)
    total = sum(math.exp(value - largest) for value in finite)
    return largest + math.log(total)


# --------------------------------------------------------------------------
# frequency handling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrequencyEstimate:
    """``P(HPO | disease)`` together with where the number came from.

    ``effective_probability`` is what the likelihood actually used;
    ``raw_value``/``kind``/``category_term`` keep the curated source visible so a
    reviewer can tell "60% of 20 reported patients" apart from "Frequent" apart
    from "nobody recorded a frequency".
    """

    effective_probability: Optional[float]
    contributes_to_likelihood: bool
    kind: str
    policy_applied: str
    raw_value: Optional[str] = None
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    category_term: Optional[str] = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "effective_probability": self.effective_probability,
            "contributes_to_likelihood": self.contributes_to_likelihood,
            "kind": self.kind,
            "policy_applied": self.policy_applied,
            "raw_value": self.raw_value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "category_term": self.category_term,
            "note": self.note,
        }


def smooth_ratio(
    numerator: float,
    denominator: float,
    *,
    config: LikelihoodConfig = DEFAULT_CONFIG,
) -> float:
    """Beta-smoothed ``n/m`` so 0/12 and 12/12 do not become certainties."""

    a = config.beta_pseudo_count
    if denominator <= 0.0:
        return clamp_probability(config.missing_frequency_default, epsilon=config.epsilon)
    return clamp_probability(
        (numerator + a) / (denominator + 2.0 * a), epsilon=config.epsilon
    )


def resolve_frequency(
    parsed: Optional[Mapping[str, Any]],
    *,
    policy: MissingFrequencyPolicy = MissingFrequencyPolicy.NEUTRAL,
    background_probability: float,
    config: LikelihoodConfig = DEFAULT_CONFIG,
    excluded: bool = False,
) -> FrequencyEstimate:
    """Turn a parsed HPOA frequency field into an effective probability.

    ``parsed`` is the mapping produced by
    :func:`hpo_agents.ontology.parse_frequency_detail`.  ``None`` or a mapping
    whose ``kind`` is ``"missing"`` means the curators recorded no frequency, and
    then ``policy`` decides what happens -- never a silent ``0.5``.
    """

    if excluded:
        # A curated NOT/Excluded annotation is information, not a missing value.
        return FrequencyEstimate(
            effective_probability=clamp_probability(config.epsilon, epsilon=config.epsilon),
            contributes_to_likelihood=True,
            kind="excluded",
            policy_applied="curated_exclusion",
            raw_value=(parsed or {}).get("raw"),
            category_term=(parsed or {}).get("category_term"),
            note="Curated NOT/Excluded annotation; smoothed away from exactly zero.",
        )

    kind = (parsed or {}).get("kind", "missing")
    raw_value = (parsed or {}).get("raw")

    if parsed is None or kind == "missing":
        if policy == MissingFrequencyPolicy.EXCLUDE_FROM_LIKELIHOOD_BUT_KEEP_SEMANTIC_MATCH:
            return FrequencyEstimate(
                effective_probability=None,
                contributes_to_likelihood=False,
                kind="missing",
                policy_applied=policy.value,
                raw_value=raw_value,
                note=(
                    "No curated frequency; the term still counts as a semantic match "
                    "but adds no likelihood evidence."
                ),
            )
        if policy == MissingFrequencyPolicy.CONFIGURABLE_DEFAULT:
            return FrequencyEstimate(
                effective_probability=clamp_probability(
                    config.missing_frequency_default, epsilon=config.epsilon
                ),
                contributes_to_likelihood=True,
                kind="missing",
                policy_applied=policy.value,
                raw_value=raw_value,
                note=(
                    "No curated frequency; the explicitly configured default "
                    f"{config.missing_frequency_default} was applied."
                ),
            )
        # NEUTRAL: match the background exactly so log LR is 0.
        return FrequencyEstimate(
            effective_probability=clamp_probability(
                background_probability, epsilon=config.epsilon
            ),
            contributes_to_likelihood=True,
            kind="missing",
            policy_applied=MissingFrequencyPolicy.NEUTRAL.value,
            raw_value=raw_value,
            note="No curated frequency; set equal to the background so log LR is 0.",
        )

    numerator = (parsed or {}).get("numerator")
    denominator = (parsed or {}).get("denominator")
    if kind == "ratio" and denominator:
        probability = smooth_ratio(float(numerator or 0.0), float(denominator), config=config)
        note = (
            f"Curated ratio {float(numerator or 0.0):g}/{float(denominator):g}, "
            f"Beta({config.beta_pseudo_count}, {config.beta_pseudo_count}) smoothed."
        )
    else:
        probability = clamp_probability(
            float((parsed or {}).get("probability", config.missing_frequency_default)),
            epsilon=config.epsilon,
        )
        note = {
            "category": "HPO frequency category mapped to a representative probability.",
            "percent": "Curated percentage.",
            "numeric": "Curated numeric frequency.",
        }.get(kind, "Curated frequency.")

    return FrequencyEstimate(
        effective_probability=probability,
        contributes_to_likelihood=True,
        kind=kind,
        policy_applied="curated_value",
        raw_value=raw_value,
        numerator=None if numerator is None else float(numerator),
        denominator=None if denominator is None else float(denominator),
        category_term=(parsed or {}).get("category_term"),
        note=note,
    )


# --------------------------------------------------------------------------
# likelihood accumulation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LikelihoodTerm:
    """One finding's bounded contribution to ``log posterior odds``.

    Kept per finding so every number in the output can be traced back to the
    HPO term, the curated frequency and the background model that produced it.
    """

    hpo_id: str
    status: str
    matched_hpo_id: Optional[str]
    relation: str
    disease_probability: Optional[float]
    background_probability: float
    background_model: BackgroundModel
    raw_log_lr: float
    capped_log_lr: float
    weight: float
    contribution: float
    was_capped: bool
    frequency: Optional[FrequencyEstimate] = None
    group_id: Optional[str] = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "hpo_id": self.hpo_id,
            "status": self.status,
            "matched_hpo_id": self.matched_hpo_id,
            "relation": self.relation,
            "disease_probability": self.disease_probability,
            "background_probability": self.background_probability,
            "background_model": self.background_model.value,
            "raw_log_lr": self.raw_log_lr,
            "capped_log_lr": self.capped_log_lr,
            "weight": self.weight,
            "contribution": self.contribution,
            "was_capped": self.was_capped,
            "frequency": self.frequency.as_dict() if self.frequency else None,
            "group_id": self.group_id,
            "note": self.note,
        }


def present_log_lr(
    disease_probability: float,
    background_probability: float,
    *,
    config: LikelihoodConfig = DEFAULT_CONFIG,
) -> float:
    """``log P(HPO | D) - log P(HPO | not D)`` for an observed finding."""

    p = clamp_probability(disease_probability, epsilon=config.epsilon)
    q = clamp_probability(background_probability, epsilon=config.epsilon)
    return math.log(p) - math.log(q)


def absent_log_lr(
    disease_probability: float,
    background_probability: float,
    *,
    config: LikelihoodConfig = DEFAULT_CONFIG,
) -> float:
    """``log(1 - P(HPO | D)) - log(1 - P(HPO | not D))`` for a confirmed absence."""

    p = clamp_probability(disease_probability, epsilon=config.epsilon)
    q = clamp_probability(background_probability, epsilon=config.epsilon)
    return math.log1p(-p) - math.log1p(-q)


def build_likelihood_term(
    *,
    hpo_id: str,
    status: str,
    matched_hpo_id: Optional[str],
    relation: str,
    frequency: FrequencyEstimate,
    background_probability: float,
    background_model: BackgroundModel,
    weight: float,
    config: LikelihoodConfig = DEFAULT_CONFIG,
    group_id: Optional[str] = None,
    note: str = "",
) -> LikelihoodTerm:
    """Assemble one bounded, explainable likelihood term."""

    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must lie in [0, 1]; got {weight!r}")
    background = clamp_probability(background_probability, epsilon=config.epsilon)

    if not frequency.contributes_to_likelihood or frequency.effective_probability is None:
        return LikelihoodTerm(
            hpo_id=hpo_id,
            status=status,
            matched_hpo_id=matched_hpo_id,
            relation=relation,
            disease_probability=None,
            background_probability=background,
            background_model=background_model,
            raw_log_lr=0.0,
            capped_log_lr=0.0,
            weight=float(weight),
            contribution=0.0,
            was_capped=False,
            frequency=frequency,
            group_id=group_id,
            note=note or frequency.note,
        )

    disease_probability = clamp_probability(
        frequency.effective_probability, epsilon=config.epsilon
    )
    if status == "ABSENT":
        raw = absent_log_lr(disease_probability, background, config=config)
    else:
        raw = present_log_lr(disease_probability, background, config=config)

    bound = config.max_abs_log_lr_per_term
    capped = min(bound, max(-bound, raw))
    return LikelihoodTerm(
        hpo_id=hpo_id,
        status=status,
        matched_hpo_id=matched_hpo_id,
        relation=relation,
        disease_probability=disease_probability,
        background_probability=background,
        background_model=background_model,
        raw_log_lr=float(raw),
        capped_log_lr=float(capped),
        weight=float(weight),
        contribution=float(weight * capped),
        was_capped=bool(abs(raw - capped) > 1e-12),
        frequency=frequency,
        group_id=group_id,
        note=note or frequency.note,
    )


@dataclass(frozen=True)
class LogLikelihoodResult:
    """Total ``log LR`` for one disease plus the terms that produced it."""

    log_likelihood_ratio: float
    terms: tuple[LikelihoodTerm, ...]
    capped_term_count: int
    capped_group_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "log_likelihood_ratio": self.log_likelihood_ratio,
            "terms": [term.as_dict() for term in self.terms],
            "capped_term_count": self.capped_term_count,
            "capped_group_ids": list(self.capped_group_ids),
        }


def combine_log_likelihood(
    terms: Sequence[LikelihoodTerm],
    *,
    config: LikelihoodConfig = DEFAULT_CONFIG,
) -> LogLikelihoodResult:
    """Sum term contributions, optionally capping each dependent group.

    Findings that share an organ-system ancestor are not independent, so their
    joint contribution is optionally bounded by
    ``config.max_abs_log_lr_per_group``.  This does not make the naive
    conditional-independence assumption correct; it limits how wrong it can be.
    """

    grouped: dict[Optional[str], list[LikelihoodTerm]] = {}
    for term in terms:
        grouped.setdefault(term.group_id, []).append(term)

    total = 0.0
    capped_groups: list[str] = []
    for group_id, group_terms in grouped.items():
        subtotal = math.fsum(term.contribution for term in group_terms)
        limit = config.max_abs_log_lr_per_group
        if group_id is not None and limit is not None and abs(subtotal) > limit:
            subtotal = math.copysign(limit, subtotal)
            capped_groups.append(group_id)
        total += subtotal

    if not math.isfinite(total):
        raise ValueError("accumulated log likelihood ratio is not finite")
    return LogLikelihoodResult(
        log_likelihood_ratio=float(total),
        terms=tuple(terms),
        capped_term_count=sum(1 for term in terms if term.was_capped),
        capped_group_ids=tuple(sorted(capped_groups)),
    )


# --------------------------------------------------------------------------
# posteriors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorEstimate:
    """A probability *plus* the semantics needed to read it correctly."""

    probability: Optional[float]
    kind: ProbabilityKind
    scope: ProbabilityScope
    calibrated: bool
    prior: Optional[DiseasePrior] = None
    log_prior_odds: Optional[float] = None
    log_likelihood_ratio: float = 0.0
    log_posterior_odds: Optional[float] = None
    calibrator_metadata: Optional[Mapping[str, Any]] = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.probability is not None:
            value = float(self.probability)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"posterior probability must lie in [0, 1]; got {value!r}")
        if self.probability is None and self.kind != ProbabilityKind.RANKING_SCORE:
            raise ValueError("only RANKING_SCORE may omit a probability")
        if self.probability is not None and self.kind == ProbabilityKind.RANKING_SCORE:
            raise ValueError("RANKING_SCORE must not carry an absolute probability")
        if self.calibrated and self.kind != ProbabilityKind.CALIBRATED_PROBABILITY:
            raise ValueError("calibrated=True requires kind=CALIBRATED_PROBABILITY")

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimated_probability": self.probability,
            "probability_kind": self.kind.value,
            "probability_scope": self.scope.value,
            "calibrated": self.calibrated,
            "prior": self.prior.as_dict() if self.prior else None,
            "log_prior_odds": self.log_prior_odds,
            "log_likelihood_ratio": self.log_likelihood_ratio,
            "log_posterior_odds": self.log_posterior_odds,
            "calibrator_metadata": (
                dict(self.calibrator_metadata) if self.calibrator_metadata else None
            ),
            "notes": list(self.notes),
        }


def ranking_only(
    log_likelihood_ratio: float,
    *,
    note: str = "No disease prior was supplied, so only a ranking score is reported.",
) -> PosteriorEstimate:
    """The honest answer when there is no prior: rank, do not quantify."""

    return PosteriorEstimate(
        probability=None,
        kind=ProbabilityKind.RANKING_SCORE,
        scope=ProbabilityScope.NOT_APPLICABLE,
        calibrated=False,
        prior=None,
        log_prior_odds=None,
        log_likelihood_ratio=float(log_likelihood_ratio),
        log_posterior_odds=None,
        notes=(note,),
    )


def binary_posterior(
    prior: DiseasePrior,
    log_likelihood_ratio: float,
    *,
    config: LikelihoodConfig = DEFAULT_CONFIG,
    notes: Sequence[str] = (),
) -> PosteriorEstimate:
    """``P(disease | evidence)`` for one disease considered on its own.

    These are per-disease binary posteriors: two diseases can both be likely,
    and the values across a differential are not required to sum to one.
    """

    prior_log_odds = log_odds(prior.prior_probability)
    total = prior_log_odds + float(log_likelihood_ratio)
    if not math.isfinite(total):
        raise ValueError("posterior log-odds is not finite")
    probability = probability_from_log_odds(total)
    floor = config.reported_probability_floor
    probability = min(1.0 - floor, max(floor, probability))
    return PosteriorEstimate(
        probability=float(probability),
        kind=ProbabilityKind.BAYES_ESTIMATE_UNCALIBRATED,
        scope=ProbabilityScope.BINARY_PER_DISEASE,
        calibrated=False,
        prior=prior,
        log_prior_odds=float(prior_log_odds),
        log_likelihood_ratio=float(log_likelihood_ratio),
        log_posterior_odds=float(total),
        notes=tuple(notes)
        or (
            "Uncalibrated Bayes estimate under the stated prior, background model "
            "and naive conditional independence; not a validated clinical probability.",
        ),
    )


@dataclass(frozen=True)
class SingleCauseEntry:
    """One candidate in a differential that assumes a single principal cause."""

    disease_id: str
    log_likelihood_ratio: float
    prior_probability: float
    disease_name: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.log_likelihood_ratio)):
            raise ValueError(f"log_likelihood_ratio for {self.disease_id} is not finite")
        probability = float(self.prior_probability)
        if not math.isfinite(probability) or probability <= 0.0 or probability >= 1.0:
            raise ValueError(
                f"prior_probability for {self.disease_id} must satisfy 0 < p < 1; "
                f"got {self.prior_probability!r}"
            )


@dataclass(frozen=True)
class UnknownClass:
    """The mandatory "none of the above" class of a single-cause differential.

    Without it, normalising over a handful of curated diseases would force the
    probabilities to sum to one even when none of them fits the findings.
    """

    prior_probability: float
    log_likelihood_ratio: float = 0.0
    disease_id: str = UNKNOWN_DISEASE_ID
    disease_name: str = UNKNOWN_DISEASE_NAME

    def __post_init__(self) -> None:
        probability = float(self.prior_probability)
        if not math.isfinite(probability) or probability <= 0.0 or probability >= 1.0:
            raise ValueError(
                f"unknown prior_probability must satisfy 0 < p < 1; got {self.prior_probability!r}"
            )
        if not math.isfinite(float(self.log_likelihood_ratio)):
            raise ValueError("unknown log_likelihood_ratio must be finite")


def derive_unknown_prior(
    candidate_priors: Sequence[float],
    *,
    minimum: float = 0.05,
) -> float:
    """Left-over probability mass for the unknown class.

    Defaults to ``1 - sum(candidate priors)`` but never drops below ``minimum``:
    a shortlist of curated diseases is not a closed world, so the unknown class
    always keeps some mass.
    """

    if not 0.0 < minimum < 1.0:
        raise ValueError("minimum unknown prior must satisfy 0 < minimum < 1")
    remainder = 1.0 - math.fsum(float(value) for value in candidate_priors)
    return float(max(minimum, min(1.0 - minimum, remainder)))


@dataclass(frozen=True)
class SingleCauseResult:
    """Relative probabilities within one candidate set plus the unknown class."""

    probabilities: Mapping[str, float]
    log_weights: Mapping[str, float]
    unknown_probability: float
    top_disease_id: str
    unknown_dominates: bool
    scope: ProbabilityScope = ProbabilityScope.SINGLE_CAUSE_CANDIDATE_SET
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "probabilities": dict(self.probabilities),
            "log_weights": dict(self.log_weights),
            "unknown_probability": self.unknown_probability,
            "top_disease_id": self.top_disease_id,
            "unknown_dominates": self.unknown_dominates,
            "probability_scope": self.scope.value,
            "notes": list(self.notes),
        }


def single_cause_posteriors(
    entries: Sequence[SingleCauseEntry],
    unknown: UnknownClass,
    *,
    config: LikelihoodConfig = DEFAULT_CONFIG,
) -> SingleCauseResult:
    """Normalise ``prior x LR`` over a candidate set that includes ``unknown``.

    Uses log-sum-exp, so the result is stable however extreme the individual
    likelihood ratios are.  The output answers "given that exactly one of these
    explains the findings, which one?", which is a different question from
    :func:`binary_posterior` and is labelled with a different scope.
    """

    if not entries:
        raise ValueError("single_cause_posteriors requires at least one candidate")

    log_weights: dict[str, float] = {}
    for entry in entries:
        if entry.disease_id == unknown.disease_id:
            raise ValueError(
                f"candidate id {entry.disease_id!r} collides with the unknown class id"
            )
        log_weights[entry.disease_id] = math.log(entry.prior_probability) + float(
            entry.log_likelihood_ratio
        )
    log_weights[unknown.disease_id] = math.log(unknown.prior_probability) + float(
        unknown.log_likelihood_ratio
    )

    normaliser = log_sum_exp(list(log_weights.values()))
    probabilities = {
        disease_id: math.exp(value - normaliser) for disease_id, value in log_weights.items()
    }
    for value in probabilities.values():
        if not math.isfinite(value):
            raise ValueError("single-cause normalisation produced a non-finite probability")

    unknown_probability = probabilities[unknown.disease_id]
    top_disease_id = max(probabilities, key=lambda key: (probabilities[key], key))
    return SingleCauseResult(
        probabilities=probabilities,
        log_weights=log_weights,
        unknown_probability=float(unknown_probability),
        top_disease_id=top_disease_id,
        unknown_dominates=bool(unknown_probability >= config.unknown_dominance_threshold),
        notes=(
            "Relative probability within this candidate set under an explicit "
            "single-principal-cause assumption; includes an explicit "
            f"{unknown.disease_id} class and is not a per-disease probability.",
        ),
    )


def apply_calibration(
    estimate: PosteriorEstimate,
    calibrated_probability: float,
    calibrator_metadata: Mapping[str, Any],
    *,
    config: LikelihoodConfig = DEFAULT_CONFIG,
) -> PosteriorEstimate:
    """Promote a Bayes estimate to a calibrated probability.

    The caller is responsible for having verified that the calibrator is
    compatible and clinically usable; this function only rewrites the labels so
    the two kinds can never be confused downstream.
    """

    value = float(calibrated_probability)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"calibrated probability must lie in [0, 1]; got {calibrated_probability!r}")
    floor = config.reported_probability_floor
    return PosteriorEstimate(
        probability=min(1.0 - floor, max(floor, value)),
        kind=ProbabilityKind.CALIBRATED_PROBABILITY,
        scope=estimate.scope,
        calibrated=True,
        prior=estimate.prior,
        log_prior_odds=estimate.log_prior_odds,
        log_likelihood_ratio=estimate.log_likelihood_ratio,
        log_posterior_odds=estimate.log_posterior_odds,
        calibrator_metadata=dict(calibrator_metadata),
        notes=estimate.notes,
    )


__all__ = [
    "DEFAULT_CONFIG",
    "UNKNOWN_DISEASE_ID",
    "UNKNOWN_DISEASE_NAME",
    "FrequencyEstimate",
    "LikelihoodConfig",
    "LikelihoodTerm",
    "LogLikelihoodResult",
    "PosteriorEstimate",
    "SingleCauseEntry",
    "SingleCauseResult",
    "UnknownClass",
    "absent_log_lr",
    "apply_calibration",
    "binary_posterior",
    "build_likelihood_term",
    "clamp_probability",
    "combine_log_likelihood",
    "derive_unknown_prior",
    "log_odds",
    "log_sum_exp",
    "present_log_lr",
    "probability_from_log_odds",
    "ranking_only",
    "resolve_frequency",
    "smooth_ratio",
]
