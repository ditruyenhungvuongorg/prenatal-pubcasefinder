"""Safe, evidence-grounded ranking for Agent 3.

The component intentionally separates four concerns:

* reranking disease candidates already produced by Agent 2;
* ranking genes from explicit disease-gene associations;
* emitting transparent, rule-based test *considerations*;
* ranking variants only when caller-supplied VCF records are present.

It is a decision-support component.  It does not diagnose disease, manufacture
evidence, or infer a variant from phenotype data alone.
"""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping, Sequence


COMPONENT_VERSION = "1.1.0"


class VariantDataRequiredError(ValueError):
    """Raised when variant ranking is requested without non-empty VCF data."""


class _NumpyLogisticClassifier:
    """Small deterministic binary classifier used when sklearn is unavailable."""

    def __init__(self, *, iterations: int = 1800, learning_rate: float = 0.06, l2: float = 1e-3) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2 = l2
        self.mean_: Any = None
        self.scale_: Any = None
        self.coef_: Any = None
        self.intercept_: float = 0.0

    def fit(self, x: Any, y: Any) -> "_NumpyLogisticClassifier":
        import numpy as np

        matrix = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float)
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        normalized = (matrix - self.mean_) / self.scale_
        self.coef_ = np.zeros(normalized.shape[1], dtype=float)
        for _ in range(self.iterations):
            logits = np.clip(normalized @ self.coef_ + self.intercept_, -35.0, 35.0)
            predictions = 1.0 / (1.0 + np.exp(-logits))
            residual = predictions - target
            self.coef_ -= self.learning_rate * (
                normalized.T @ residual / len(target) + self.l2 * self.coef_
            )
            self.intercept_ -= self.learning_rate * float(residual.mean())
        return self

    def predict_proba(self, x: Any) -> Any:
        import numpy as np

        matrix = np.asarray(x, dtype=float)
        normalized = (matrix - self.mean_) / self.scale_
        logits = np.clip(normalized @ self.coef_ + self.intercept_, -35.0, 35.0)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack((1.0 - positive, positive))


class Agent3Ranker:
    """Rank diseases and genes while keeping evidence and provenance explicit.

    Parameters
    ----------
    disease_gene_associations:
        Optional default association collection.  Each row should contain
        ``disease_id``, ``gene_symbol``, ``evidence_type`` and provenance such
        as ``source`` / ``source_version``.
    reranker_backend:
        ``"deterministic"`` (default), ``"auto"``, ``"logistic"``,
        ``"hist_gradient_boosting"``, ``"xgboost"`` or ``"numpy_logistic"``.
        Trainable backends are
        activated only after :meth:`fit`; otherwise ranking safely falls back to
        the deterministic Agent-2 score order.
    top_diseases / top_genes:
        Hard rank cutoffs for disease-to-gene propagation and retained genes.
    minimum_association_weight / minimum_gene_score:
        Evidence gates applied before a gene can enter the ranked output.
    minimum_test_association_weight:
        Stricter association-evidence gate for emitting a test consideration.
    """

    FEATURE_NAMES = (
        "agent2_score",
        "matched_hpo_count",
        "conflicting_hpo_count",
        "unassessed_hpo_count",
        "supporting_evidence_count",
    )

    ASSOCIATION_WEIGHTS = {
        "confirmed_molecular": 1.00,
        "pathogenic_variant": 1.00,
        "curated": 0.90,
        "strong": 0.80,
        "moderate": 0.60,
        "limited": 0.35,
        "candidate": 0.20,
    }

    # These are transparent routing rules, not an ML model.  They only fire
    # when the association input explicitly contains ``test_evidence_type``.
    TEST_RULES = {
        "chromosomal": (
            "karyotype",
            "A chromosomal evidence type was supplied in the association data.",
        ),
        "copy_number": (
            "CMA",
            "A copy-number evidence type was supplied in the association data.",
        ),
        "cnv": (
            "CMA",
            "A CNV evidence type was supplied in the association data.",
        ),
        "single_gene": (
            "targeted_gene_test",
            "A single-gene evidence type was supplied in the association data.",
        ),
        "sequence_variant": (
            "sequence_based_test",
            "A sequence-variant evidence type was supplied in the association data.",
        ),
        "exome": (
            "WES",
            "An exome evidence type was supplied in the association data.",
        ),
        "genome": (
            "WGS",
            "A genome evidence type was supplied in the association data.",
        ),
    }

    def __init__(
        self,
        disease_gene_associations: Sequence[Mapping[str, Any]]
        | Mapping[str, Any]
        | None = None,
        *,
        reranker_backend: str = "deterministic",
        component_version: str = COMPONENT_VERSION,
        association_version: str | None = None,
        top_diseases: int = 10,
        top_genes: int = 20,
        minimum_association_weight: float = 0.35,
        minimum_gene_score: float = 0.05,
        minimum_test_association_weight: float = 0.60,
    ) -> None:
        allowed = {
            "deterministic",
            "auto",
            "logistic",
            "hist_gradient_boosting",
            "xgboost",
            "numpy_logistic",
        }
        if reranker_backend not in allowed:
            raise ValueError(
                f"Unsupported reranker_backend={reranker_backend!r}; "
                f"choose one of {sorted(allowed)}"
            )
        if (
            isinstance(top_diseases, bool)
            or not isinstance(top_diseases, int)
            or top_diseases < 1
        ):
            raise ValueError("top_diseases must be a positive integer")
        if (
            isinstance(top_genes, bool)
            or not isinstance(top_genes, int)
            or top_genes < 1
        ):
            raise ValueError("top_genes must be a positive integer")
        for name, value, maximum in (
            ("minimum_association_weight", minimum_association_weight, 1.0),
            ("minimum_gene_score", minimum_gene_score, None),
            (
                "minimum_test_association_weight",
                minimum_test_association_weight,
                1.0,
            ),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a finite non-negative number")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must be a finite non-negative number"
                ) from exc
            if not math.isfinite(numeric_value) or numeric_value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
            if maximum is not None and numeric_value > maximum:
                raise ValueError(f"{name} must be between 0 and {maximum:g}")
        self.reranker_backend = reranker_backend
        self.component_version = component_version
        self.association_version = association_version
        self.top_diseases = top_diseases
        self.top_genes = top_genes
        self.minimum_association_weight = float(minimum_association_weight)
        self.minimum_gene_score = float(minimum_gene_score)
        self.minimum_test_association_weight = float(
            minimum_test_association_weight
        )
        self._default_associations = self._normalise_associations(
            disease_gene_associations
        )
        self._model: Any | None = None
        self._active_backend = "deterministic"
        self._fit_report: dict[str, Any] = {
            "trained": False,
            "backend": "deterministic",
            "reason": "fit_not_called",
            "feature_names": list(self.FEATURE_NAMES),
        }

    @property
    def fit_report(self) -> dict[str, Any]:
        """Return a defensive copy of the last training/fallback report."""

        return deepcopy(self._fit_report)

    def fit(
        self,
        training_examples: Sequence[Mapping[str, Any]],
        *,
        label_key: str = "label",
    ) -> dict[str, Any]:
        """Fit an optional disease reranker or report deterministic fallback.

        A training row may be a disease candidate directly or may place it in a
        ``candidate`` field.  It must also contain a binary label.  Missing
        packages, insufficient rows, invalid examples and one-class data cause
        an explicit deterministic fallback instead of a failed pipeline.
        """

        if self.reranker_backend == "deterministic":
            return self._set_fallback("deterministic_backend_requested")

        x_rows: list[list[float]] = []
        y_rows: list[int] = []
        skipped = 0
        for index, row in enumerate(training_examples):
            candidate_raw = row.get("candidate", row)
            candidate, _ = self._normalise_candidate(candidate_raw, index)
            label = row.get(label_key)
            if candidate is None or label not in (0, 1, False, True):
                skipped += 1
                continue
            x_rows.append(self._feature_vector(candidate))
            y_rows.append(int(label))

        if len(x_rows) < 4:
            return self._set_fallback(
                "insufficient_training_examples", skipped_examples=skipped
            )
        if len(set(y_rows)) < 2:
            return self._set_fallback(
                "training_labels_have_one_class", skipped_examples=skipped
            )

        if self.reranker_backend == "auto":
            backends = ["hist_gradient_boosting", "logistic", "xgboost", "numpy_logistic"]
        elif self.reranker_backend == "numpy_logistic":
            backends = ["numpy_logistic"]
        else:
            backends = [self.reranker_backend, "numpy_logistic"]
        errors: list[str] = []
        for backend in backends:
            try:
                model = self._build_model(backend)
                model.fit(x_rows, y_rows)
                self._model = model
                self._active_backend = backend
                self._fit_report = {
                    "trained": True,
                    "backend": backend,
                    "reason": None,
                    "training_examples": len(x_rows),
                    "skipped_examples": skipped,
                    "feature_names": list(self.FEATURE_NAMES),
                }
                return self.fit_report
            except (ImportError, ModuleNotFoundError) as exc:
                errors.append(f"{backend}: dependency unavailable ({exc})")
            except Exception as exc:  # model-specific failures become fallback
                errors.append(f"{backend}: training failed ({type(exc).__name__})")

        return self._set_fallback(
            "trainable_backend_unavailable_or_failed",
            skipped_examples=skipped,
            backend_errors=errors,
        )

    def rank(
        self,
        disease_candidates: Sequence[Any] | None,
        disease_gene_associations: Sequence[Mapping[str, Any]]
        | Mapping[str, Any]
        | None = None,
        *,
        case_id: str | None = None,
        ontology_version: str | None = None,
        association_version: str | None = None,
        model_version: str | None = None,
        request_variant_ranking: bool = False,
        vcf_records: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run disease, gene and rule-based test recommendation stages.

        ``request_variant_ranking=True`` requires a non-empty ``vcf_records``
        collection.  Phenotype-only input can rank genes, never variants.
        """

        if request_variant_ranking and not vcf_records:
            raise VariantDataRequiredError(
                "Variant ranking refused: provide non-empty caller-supplied "
                "VCF records. Phenotype/disease evidence alone is insufficient."
            )

        provenance = self._provenance(
            ontology_version=ontology_version,
            association_version=association_version,
            model_version=model_version,
        )
        candidates, warnings = self._prepare_candidates(disease_candidates or [])
        if not candidates:
            reason = (
                "missing_disease_candidates"
                if not disease_candidates
                else "missing_supported_disease_evidence"
            )
            return self._abstain_result(
                reason=reason,
                case_id=case_id,
                provenance=provenance,
                warnings=warnings,
            )

        disease_items, rerank_warning = self._rank_diseases(candidates)
        if rerank_warning:
            warnings.append(rerank_warning)
        # Inference can explicitly fall back from a trained backend. Refresh
        # provenance after ranking so the recorded backend matches the result.
        provenance = self._provenance(
            ontology_version=ontology_version,
            association_version=association_version,
            model_version=model_version,
        )

        associations = (
            self._normalise_associations(disease_gene_associations)
            if disease_gene_associations is not None
            else deepcopy(self._default_associations)
        )
        gene_stage = self._rank_genes(disease_items, associations)
        phenotypes_to_collect = self._collect_phenotypes(disease_items)
        test_stage = self._recommend_tests(gene_stage)
        variant_stage = (
            self.rank_variants(vcf_records or [], gene_stage)
            if request_variant_ranking
            else {
                "status": "not_requested",
                "items": [],
                "reason": None,
                "decision_support_only": True,
            }
        )

        partial_reasons: list[str] = []
        for stage in (gene_stage, test_stage, variant_stage):
            if stage.get("status") == "abstain" and stage.get("reason"):
                partial_reasons.append(str(stage["reason"]))

        return {
            "case_id": case_id,
            "status": "partial" if partial_reasons else "ok",
            "abstain_reasons": partial_reasons,
            "disease_ranking": {
                "status": "ok",
                "items": disease_items,
                "reason": None,
                "backend": self._active_backend,
            },
            "gene_ranking": gene_stage,
            "variant_ranking": variant_stage,
            "test_recommendations": test_stage,
            "phenotypes_to_collect": phenotypes_to_collect,
            "warnings": warnings,
            "provenance": provenance,
            "decision_support_only": True,
            "diagnostic_claim": False,
            "safety_message": (
                "For specialist review only; rankings and test considerations "
                "are not a diagnosis."
            ),
        }

    run = rank

    def rank_variants(
        self,
        vcf_records: Sequence[Mapping[str, Any]],
        gene_ranking: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Rank caller-supplied variants; refuse phenotype-only invocation."""

        if not vcf_records:
            raise VariantDataRequiredError(
                "Variant ranking refused: no VCF records were supplied."
            )

        if isinstance(gene_ranking, Mapping):
            gene_items = list(gene_ranking.get("items", []))
        else:
            gene_items = list(gene_ranking)
        gene_rank = {
            str(item.get("gene_symbol", "")).upper(): int(item["rank"])
            for item in gene_items
            if item.get("gene_symbol") and item.get("rank")
        }

        ranked: list[dict[str, Any]] = []
        skipped = 0
        skipped_reasons: defaultdict[str, int] = defaultdict(int)
        for index, record in enumerate(vcf_records):
            if not isinstance(record, Mapping):
                skipped += 1
                skipped_reasons["record_not_mapping"] += 1
                continue
            filter_passed, filter_value = self._variant_filter_passed(record)
            if not filter_passed:
                skipped += 1
                skipped_reasons["filter_failed"] += 1
                continue
            variant_id = record.get("variant_id") or record.get("id")
            gene = str(record.get("gene_symbol") or record.get("gene") or "").upper()
            provided_score, score_reason = self._validated_variant_score(
                record
            )
            if not variant_id:
                skipped += 1
                skipped_reasons["missing_variant_id"] += 1
                continue
            if not gene:
                skipped += 1
                skipped_reasons["missing_gene_symbol"] += 1
                continue
            if provided_score is None:
                skipped += 1
                skipped_reasons[score_reason or "missing_variant_score"] += 1
                continue
            reciprocal_gene_rank = 1.0 / gene_rank[gene] if gene in gene_rank else 0.0
            ranking_score = 0.5 * provided_score + 0.5 * reciprocal_gene_rank
            ranked.append(
                {
                    "variant_id": str(variant_id),
                    "gene_symbol": gene,
                    "ranking_score": ranking_score,
                    "provided_variant_score": provided_score,
                    "gene_rank": gene_rank.get(gene),
                    "explanations": [
                        {
                            "kind": "caller_supplied_variant_evidence",
                            "variant_id": str(variant_id),
                            "gene_symbol": gene,
                            "provided_variant_score": provided_score,
                            "classification": record.get("classification"),
                            "filter": deepcopy(filter_value),
                            "source": record.get("source"),
                            "evidence": deepcopy(record.get("evidence")),
                        }
                    ],
                }
            )

        if not ranked:
            return {
                "status": "abstain",
                "reason": "vcf_records_lack_rankable_variant_evidence",
                "items": [],
                "skipped_records": skipped,
                "skipped_record_reasons": dict(sorted(skipped_reasons.items())),
                "decision_support_only": True,
                "diagnostic_claim": False,
            }

        ranked.sort(key=lambda row: (-row["ranking_score"], row["variant_id"]))
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
        return {
            "status": "ok",
            "reason": None,
            "items": ranked,
            "skipped_records": skipped,
            "skipped_record_reasons": dict(sorted(skipped_reasons.items())),
            "decision_support_only": True,
            "diagnostic_claim": False,
        }

    def _build_model(self, backend: str) -> Any:
        if backend == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=100,
                random_state=0,
            )
        if backend == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=0,
            )
        if backend == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=0,
                eval_metric="logloss",
            )
        if backend == "numpy_logistic":
            return _NumpyLogisticClassifier()
        raise ValueError(f"Unsupported trainable backend: {backend}")

    def _set_fallback(self, reason: str, **details: Any) -> dict[str, Any]:
        self._model = None
        self._active_backend = "deterministic"
        self._fit_report = {
            "trained": False,
            "backend": "deterministic",
            "reason": reason,
            "feature_names": list(self.FEATURE_NAMES),
            **details,
        }
        return self.fit_report

    def _prepare_candidates(
        self, disease_candidates: Sequence[Any]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        prepared: list[dict[str, Any]] = []
        warnings: list[str] = []
        for index, raw in enumerate(disease_candidates):
            candidate, reason = self._normalise_candidate(raw, index)
            if candidate is None:
                warnings.append(f"candidate[{index}] skipped: {reason}")
                continue
            if not candidate["supporting_evidence"]:
                warnings.append(
                    f"candidate[{index}] skipped: no supporting evidence supplied"
                )
                continue
            prepared.append(candidate)
        return prepared, warnings

    def _normalise_candidate(
        self, raw: Any, index: int
    ) -> tuple[dict[str, Any] | None, str | None]:
        raw = self._coerce_mapping(raw)
        if raw is None:
            return None, "not a mapping/dataclass and has no as_dict()"
        disease_id = raw.get("disease_id") or raw.get("id")
        score = self._first_number(
            raw,
            ("confidence_score", "compatibility_score", "agent2_score", "score", "ranking_score"),
        )
        if not disease_id:
            return None, "missing disease_id"
        if score is None:
            return None, "missing finite Agent-2 score"

        probability_matched = self._as_list(raw.get("matched_evidence"))
        probability_conflicts = self._as_list(raw.get("conflicting_evidence"))
        matched = self._as_list(raw.get("matched_hpo")) or probability_matched
        conflicts = self._as_list(raw.get("conflicting_hpo")) or probability_conflicts
        unassessed = self._as_list(raw.get("unassessed_hpo"))
        explicit_evidence = self._as_list(raw.get("evidence"))
        matched, conflicts, unassessed = self._derive_hpo_buckets(
            explicit_evidence,
            matched=matched,
            conflicts=conflicts,
            unassessed=unassessed,
        )
        supporting = [
            self._evidence_item(item, "agent2_evidence", "agent2.evidence")
            for item in explicit_evidence
        ]
        if not supporting:
            supporting.extend(
                self._evidence_item(
                    item,
                    "phenotype_match",
                    "agent2.matched_evidence",
                )
                for item in probability_matched
            )
        supporting.extend(
            self._evidence_item(item, "phenotype_match", "agent2.matched_hpo")
            for item in matched
            if item not in probability_matched
        )
        conflict_evidence = [
            self._evidence_item(item, "phenotype_conflict", "agent2.conflicting_hpo")
            for item in conflicts
        ]
        if probability_conflicts:
            conflict_evidence = [
                self._evidence_item(
                    item,
                    "phenotype_conflict",
                    "agent2.conflicting_evidence",
                )
                for item in probability_conflicts
            ]

        return (
            {
                "_index": index,
                "disease_id": str(disease_id),
                "disease_name": raw.get("disease_name") or raw.get("name"),
                "agent2_score": float(score),
                "matched_hpo": deepcopy(matched),
                "conflicting_hpo": deepcopy(conflicts),
                "unassessed_hpo": deepcopy(unassessed),
                "phenotypes_to_collect": deepcopy(
                    self._as_list(raw.get("phenotypes_to_collect"))
                ),
                "supporting_evidence": supporting,
                "conflict_evidence": conflict_evidence,
                "agent2_provenance": deepcopy(raw.get("provenance")),
            },
            None,
        )

    def _rank_diseases(
        self, candidates: Sequence[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        warning: str | None = None
        if self._model is not None:
            try:
                scores = self._model_scores(candidates)
                score_kind = "model_rerank_score"
            except Exception as exc:
                scores = [row["agent2_score"] for row in candidates]
                score_kind = "agent2_score"
                warning = (
                    "trainable reranker inference failed; deterministic fallback "
                    f"used ({type(exc).__name__})"
                )
                self._active_backend = "deterministic"
        else:
            scores = [row["agent2_score"] for row in candidates]
            score_kind = "agent2_score"

        sortable: list[tuple[dict[str, Any], float]] = list(zip(candidates, scores))
        sortable.sort(
            key=lambda pair: (
                -pair[1],
                -pair[0]["agent2_score"],
                -len(pair[0]["matched_hpo"]),
                len(pair[0]["conflicting_hpo"]),
                pair[0]["_index"],
            )
        )

        output: list[dict[str, Any]] = []
        for rank, (candidate, score) in enumerate(sortable, 1):
            output.append(
                {
                    "rank": rank,
                    "disease_id": candidate["disease_id"],
                    "disease_name": candidate["disease_name"],
                    "ranking_score": float(score),
                    "score_kind": score_kind,
                    "agent2_score": candidate["agent2_score"],
                    "matched_hpo": deepcopy(candidate["matched_hpo"]),
                    "conflicting_hpo": deepcopy(candidate["conflicting_hpo"]),
                    "unassessed_hpo": deepcopy(candidate["unassessed_hpo"]),
                    "phenotypes_to_collect": deepcopy(
                        candidate["phenotypes_to_collect"]
                    ),
                    "explanations": deepcopy(
                        candidate["supporting_evidence"]
                        + candidate["conflict_evidence"]
                    ),
                    "provenance": deepcopy(candidate["agent2_provenance"]),
                }
            )
        return output, warning

    def _model_scores(self, candidates: Sequence[dict[str, Any]]) -> list[float]:
        features = [self._feature_vector(row) for row in candidates]
        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(features)
            return [float(row[-1]) for row in probabilities]
        if hasattr(self._model, "decision_function"):
            decisions = self._model.decision_function(features)
            return [1.0 / (1.0 + math.exp(-float(value))) for value in decisions]
        predictions = self._model.predict(features)
        return [float(value) for value in predictions]

    def _feature_vector(self, candidate: Mapping[str, Any]) -> list[float]:
        return [
            float(candidate["agent2_score"]),
            float(len(candidate["matched_hpo"])),
            float(len(candidate["conflicting_hpo"])),
            float(len(candidate["unassessed_hpo"])),
            float(len(candidate["supporting_evidence"])),
        ]

    def _rank_genes(
        self,
        disease_items: Sequence[Mapping[str, Any]],
        associations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not associations:
            return {
                "status": "abstain",
                "reason": "missing_disease_gene_associations",
                "items": [],
                "cutoffs": self._decision_cutoffs(),
                "decision_support_only": True,
                "diagnostic_claim": False,
            }

        disease_lookup = {row["disease_id"]: row for row in disease_items}
        gene_scores: defaultdict[str, float] = defaultdict(float)
        gene_explanations: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        seen: set[tuple[Any, ...]] = set()
        excluded: defaultdict[str, int] = defaultdict(int)
        associations_used = 0

        for association in associations:
            disease_id = str(association.get("disease_id") or "")
            gene = str(
                association.get("gene_symbol") or association.get("gene") or ""
            ).upper()
            if disease_id not in disease_lookup:
                excluded["disease_not_in_ranked_candidates"] += 1
                continue
            if not gene:
                excluded["missing_gene_symbol"] += 1
                continue
            evidence_type = self._normalise_token(
                association.get("evidence_type") or "candidate"
            )
            duplicate_key = (
                disease_id,
                gene,
                evidence_type,
                association.get("association_id"),
                association.get("source"),
            )
            if duplicate_key in seen:
                excluded["duplicate_association"] += 1
                continue
            seen.add(duplicate_key)

            disease = disease_lookup[disease_id]
            if int(disease["rank"]) > self.top_diseases:
                excluded["disease_rank_above_cutoff"] += 1
                continue
            disease_rank_weight = 1.0 / float(disease["rank"])
            evidence_weight = self.ASSOCIATION_WEIGHTS.get(evidence_type, 0.10)
            if evidence_weight < self.minimum_association_weight:
                excluded["association_weight_below_cutoff"] += 1
                continue
            contribution = disease_rank_weight * evidence_weight
            associations_used += 1
            gene_scores[gene] += contribution
            gene_explanations[gene].append(
                {
                    "kind": "disease_gene_association",
                    "disease_id": disease_id,
                    "disease_rank": disease["rank"],
                    "association_id": association.get("association_id"),
                    "evidence_type": evidence_type,
                    "association_weight": evidence_weight,
                    "ranking_contribution": contribution,
                    "source": association.get("source"),
                    "source_version": association.get("source_version"),
                    "test_evidence_type": association.get("test_evidence_type"),
                    "inheritance": association.get("inheritance"),
                    "evidence": deepcopy(association.get("evidence")),
                }
            )

        if not gene_scores:
            return {
                "status": "abstain",
                "reason": "no_association_meets_disease_or_evidence_cutoffs",
                "items": [],
                "cutoffs": self._decision_cutoffs(),
                "filtering": {
                    "associations_seen": len(associations),
                    "associations_used": 0,
                    "excluded_by_reason": dict(sorted(excluded.items())),
                },
                "decision_support_only": True,
                "diagnostic_claim": False,
            }

        items = [
            {
                "gene_symbol": gene,
                "ranking_score": score,
                "explanations": gene_explanations[gene],
            }
            for gene, score in gene_scores.items()
        ]
        items.sort(
            key=lambda row: (
                -row["ranking_score"],
                -len(row["explanations"]),
                row["gene_symbol"],
            )
        )
        genes_before_cutoff = len(items)
        below_score = sum(
            item["ranking_score"] < self.minimum_gene_score for item in items
        )
        if below_score:
            excluded["gene_score_below_cutoff"] += below_score
        items = [
            item
            for item in items
            if item["ranking_score"] >= self.minimum_gene_score
        ]
        above_rank = max(0, len(items) - self.top_genes)
        if above_rank:
            excluded["gene_rank_above_cutoff"] += above_rank
        items = items[: self.top_genes]
        if not items:
            return {
                "status": "abstain",
                "reason": "no_gene_meets_score_cutoff",
                "items": [],
                "cutoffs": self._decision_cutoffs(),
                "filtering": {
                    "associations_seen": len(associations),
                    "associations_used": associations_used,
                    "genes_before_cutoff": genes_before_cutoff,
                    "genes_retained": 0,
                    "excluded_by_reason": dict(sorted(excluded.items())),
                },
                "decision_support_only": True,
                "diagnostic_claim": False,
            }
        for rank, item in enumerate(items, 1):
            item["rank"] = rank
        return {
            "status": "ok",
            "reason": None,
            "items": items,
            "cutoffs": self._decision_cutoffs(),
            "filtering": {
                "associations_seen": len(associations),
                "associations_used": associations_used,
                "genes_before_cutoff": genes_before_cutoff,
                "genes_retained": len(items),
                "excluded_by_reason": dict(sorted(excluded.items())),
            },
            "decision_support_only": True,
            "diagnostic_claim": False,
        }

    def _recommend_tests(self, gene_stage: Mapping[str, Any]) -> dict[str, Any]:
        if gene_stage.get("status") != "ok":
            return {
                "status": "abstain",
                "reason": "gene_ranking_unavailable_for_test_rules",
                "items": [],
                "method": "transparent_rules_v1",
                "cutoffs": self._decision_cutoffs(),
                "decision_support_only": True,
                "diagnostic_claim": False,
            }

        by_test: dict[str, dict[str, Any]] = {}
        excluded: defaultdict[str, int] = defaultdict(int)
        explicit_rules_seen = 0
        for gene in gene_stage.get("items", []):
            gene_rank = int(gene.get("rank") or 0)
            gene_score = self._first_number(gene, ("ranking_score",))
            if not gene_rank or gene_rank > self.top_genes:
                excluded["gene_rank_above_cutoff"] += 1
                continue
            if gene_score is None or gene_score < self.minimum_gene_score:
                excluded["gene_score_below_cutoff"] += 1
                continue
            for evidence in gene.get("explanations", []):
                # Test routing requires a separate, explicitly curated field.
                # A generic disease-gene association (for example "curated")
                # is not enough to recommend a laboratory modality.
                evidence_type = self._normalise_token(
                    evidence.get("test_evidence_type")
                )
                if evidence_type not in self.TEST_RULES:
                    continue
                explicit_rules_seen += 1
                disease_rank = int(evidence.get("disease_rank") or 0)
                association_weight = self._first_number(
                    evidence, ("association_weight",)
                )
                if not disease_rank or disease_rank > self.top_diseases:
                    excluded["disease_rank_above_cutoff"] += 1
                    continue
                if (
                    association_weight is None
                    or association_weight < self.minimum_test_association_weight
                ):
                    excluded["test_association_weight_below_cutoff"] += 1
                    continue
                test_name, rationale = self.TEST_RULES[evidence_type]
                item = by_test.setdefault(
                    test_name,
                    {
                        "test": test_name,
                        "rule_id": f"evidence_type:{evidence_type}",
                        "rationale": rationale,
                        "basis": [],
                        "recommendation": (
                            f"Consider {test_name} only after review by the "
                            "responsible genetics specialist."
                        ),
                        "diagnostic_claim": False,
                    },
                )
                item["basis"].append(
                    {
                        "gene_symbol": gene.get("gene_symbol"),
                        "gene_rank": gene_rank,
                        "gene_ranking_score": gene_score,
                        "disease_id": evidence.get("disease_id"),
                        "disease_rank": disease_rank,
                        "evidence_type": evidence_type,
                        "association_weight": association_weight,
                        "association_id": evidence.get("association_id"),
                        "source": evidence.get("source"),
                        "source_version": evidence.get("source_version"),
                    }
                )

        if not by_test:
            return {
                "status": "abstain",
                "reason": (
                    "test_rules_below_evidence_cutoff"
                    if explicit_rules_seen
                    else "no_supported_test_rule_for_supplied_evidence_types"
                ),
                "items": [],
                "method": "transparent_rules_v1",
                "cutoffs": self._decision_cutoffs(),
                "filtering": {
                    "explicit_rules_seen": explicit_rules_seen,
                    "rules_emitted": 0,
                    "excluded_by_reason": dict(sorted(excluded.items())),
                },
                "decision_support_only": True,
                "diagnostic_claim": False,
            }

        return {
            "status": "ok",
            "reason": None,
            "items": [by_test[key] for key in sorted(by_test)],
            "method": "transparent_rules_v1",
            "cutoffs": self._decision_cutoffs(),
            "filtering": {
                "explicit_rules_seen": explicit_rules_seen,
                "rules_emitted": sum(
                    len(item["basis"]) for item in by_test.values()
                ),
                "excluded_by_reason": dict(sorted(excluded.items())),
            },
            "decision_support_only": True,
            "diagnostic_claim": False,
        }

    def _collect_phenotypes(
        self, disease_items: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for disease in disease_items:
            sources = (
                ("agent2.unassessed_hpo", disease.get("unassessed_hpo", [])),
                (
                    "agent2.phenotypes_to_collect",
                    disease.get("phenotypes_to_collect", []),
                ),
            )
            for source, values in sources:
                for value in values:
                    key = self._phenotype_key(value)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    collected.append(
                        {
                            "phenotype": deepcopy(value),
                            "source": source,
                            "disease_id": disease.get("disease_id"),
                            "disease_rank": disease.get("rank"),
                        }
                    )
        return collected

    def _normalise_associations(
        self,
        associations: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not associations:
            return []
        output: list[dict[str, Any]] = []
        if isinstance(associations, Mapping):
            for disease_id, rows in associations.items():
                for row in self._as_list(rows):
                    if isinstance(row, str):
                        output.append(
                            {"disease_id": str(disease_id), "gene_symbol": row}
                        )
                    else:
                        item = self._coerce_mapping(row)
                        if item is None:
                            continue
                        item = dict(item)
                        item.setdefault("disease_id", str(disease_id))
                        output.append(item)
            return output
        for row in associations:
            item = self._coerce_mapping(row)
            if item is not None:
                output.append(dict(item))
        return output

    def _derive_hpo_buckets(
        self,
        evidence_items: Sequence[Any],
        *,
        matched: list[Any],
        conflicts: list[Any],
        unassessed: list[Any],
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Derive display buckets from Agent-2 EvidenceItem dictionaries.

        This does not create new medical evidence. It only reorganises fields
        already supplied by Agent 2, including its ``status``, ``relation`` and
        signed contribution.
        """

        matched_out = deepcopy(matched)
        conflicts_out = deepcopy(conflicts)
        unassessed_out = deepcopy(unassessed)
        for raw_item in evidence_items:
            item = self._coerce_mapping(raw_item)
            if item is None:
                continue
            status = self._normalise_token(item.get("status"))
            relation = self._normalise_token(item.get("relation"))
            observed = item.get("observed_hpo_id") or item.get("hpo_id")
            matched_hpo = item.get("matched_hpo_id")
            descriptor = {
                key: deepcopy(item.get(key))
                for key in ("observed_hpo_id", "matched_hpo_id", "relation", "status")
                if item.get(key) is not None
            }
            if status in {"present", "suspected"} and matched_hpo and relation not in {
                "",
                "none",
                "unmatched",
            }:
                if not matched:
                    matched_out.append(descriptor)
            elif status in {"not_assessed", "unassessed"} and observed:
                if not unassessed:
                    unassessed_out.append(str(observed))
            elif status == "absent" and matched_hpo:
                contribution = self._first_number(
                    item, ("weighted_contribution", "log_likelihood_ratio")
                )
                if contribution is not None and contribution < 0 and not conflicts:
                    conflicts_out.append(descriptor)
        return matched_out, conflicts_out, unassessed_out

    def _provenance(
        self,
        *,
        ontology_version: str | None,
        association_version: str | None,
        model_version: str | None,
    ) -> dict[str, Any]:
        return {
            "component": "hpo_agents.agent3_ranker",
            "component_version": self.component_version,
            "reranker_backend": self._active_backend,
            "model_version": model_version,
            "association_version": association_version or self.association_version,
            "ontology_version": ontology_version,
            "feature_names": list(self.FEATURE_NAMES),
            "test_recommendation_method": "transparent_rules_v1",
            "decision_cutoffs": self._decision_cutoffs(),
        }

    def _decision_cutoffs(self) -> dict[str, Any]:
        """Return the active evidence gates recorded in every downstream stage."""

        return {
            "top_diseases": self.top_diseases,
            "top_genes": self.top_genes,
            "minimum_association_weight": self.minimum_association_weight,
            "minimum_gene_score": self.minimum_gene_score,
            "minimum_test_association_weight": (
                self.minimum_test_association_weight
            ),
        }

    def _abstain_result(
        self,
        *,
        reason: str,
        case_id: str | None,
        provenance: Mapping[str, Any],
        warnings: Sequence[str],
    ) -> dict[str, Any]:
        abstain_stage = {
            "status": "abstain",
            "reason": reason,
            "items": [],
            "decision_support_only": True,
        }
        return {
            "case_id": case_id,
            "status": "abstain",
            "abstain_reasons": [reason],
            "disease_ranking": deepcopy(abstain_stage),
            "gene_ranking": deepcopy(abstain_stage),
            "variant_ranking": {
                "status": "not_requested",
                "reason": None,
                "items": [],
                "decision_support_only": True,
            },
            "test_recommendations": deepcopy(abstain_stage),
            "phenotypes_to_collect": [],
            "warnings": list(warnings),
            "provenance": dict(provenance),
            "decision_support_only": True,
            "diagnostic_claim": False,
            "safety_message": (
                "Agent 3 abstained because required evidence was missing; no "
                "clinical conclusion was produced."
            ),
        }

    @staticmethod
    def _evidence_item(item: Any, kind: str, source: str) -> dict[str, Any]:
        if isinstance(item, Mapping):
            result = deepcopy(dict(item))
            result.setdefault("kind", kind)
            result.setdefault("source", source)
            return result
        return {"kind": kind, "value": deepcopy(item), "source": source}

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any] | None:
        if isinstance(value, Mapping):
            return dict(value)
        as_dict_method = getattr(value, "as_dict", None)
        if callable(as_dict_method):
            converted = as_dict_method()
            if isinstance(converted, Mapping):
                return dict(converted)
        if is_dataclass(value):
            converted = asdict(value)
            if isinstance(converted, Mapping):
                return dict(converted)
        return None

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (tuple, set)):
            return list(value)
        return [value]

    @staticmethod
    def _first_number(
        row: Mapping[str, Any], keys: Iterable[str]
    ) -> float | None:
        for key in keys:
            value = row.get(key)
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
        return None

    @staticmethod
    def _validated_variant_score(
        row: Mapping[str, Any],
    ) -> tuple[float | None, str | None]:
        """Validate the highest-priority supplied score without silent clamping."""

        for key in ("variant_score", "pathogenicity_score", "score"):
            if key not in row or row.get(key) in (None, ""):
                continue
            value = row.get(key)
            if isinstance(value, bool):
                return None, "invalid_variant_score"
            try:
                score = float(value)
            except (TypeError, ValueError):
                return None, "invalid_variant_score"
            if not math.isfinite(score):
                return None, "non_finite_variant_score"
            if not 0.0 <= score <= 1.0:
                return None, "variant_score_out_of_range"
            return score, None
        return None, "missing_variant_score"

    @staticmethod
    def _variant_filter_passed(
        row: Mapping[str, Any],
    ) -> tuple[bool, Any]:
        """Treat absent/PASS VCF filters as usable and fail closed otherwise."""

        found = False
        recorded_value: Any = None
        for key in ("FILTER", "filter", "filt"):
            if key not in row:
                continue
            found = True
            value = row.get(key)
            recorded_value = deepcopy(value)
            values = value if isinstance(value, (list, tuple, set)) else [value]
            tokens: list[str] = []
            for item in values:
                if item is None:
                    tokens.append("")
                else:
                    tokens.extend(str(item).strip().split(";"))
            if any(
                token.strip().upper() not in {"", ".", "PASS"}
                for token in tokens
            ):
                return False, recorded_value
        return True, recorded_value if found else None

    @staticmethod
    def _normalise_token(value: Any) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _phenotype_key(value: Any) -> str:
        if isinstance(value, Mapping):
            for key in ("hpo_id", "id", "label", "name", "value"):
                if value.get(key):
                    return str(value[key])
            return ""
        return str(value) if value is not None else ""


__all__ = ["Agent3Ranker", "VariantDataRequiredError", "COMPONENT_VERSION"]
