"""Orchestration for the three-agent research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .agent1_contract import profile_to_agent2_request
from .agent2_schema import ClinicalContext, PhenotypeObservation
from .agent3_prenatal_recommender import PrenatalDecisionAgent
from .agent3_ranker import Agent3Ranker
from .review import require_clinician_approval
from .schemas import PatientProfile, jsonable


@dataclass(slots=True)
class ThreeAgentPipeline:
    agent1: Any
    agent2: Any
    agent3: Agent3Ranker

    def extract(self, text: str, *, case_id: str = "CASE", **kwargs: Any) -> PatientProfile:
        profile = self.agent1.extract(text, case_id=case_id, **kwargs)
        if not isinstance(profile, PatientProfile):
            raise TypeError("Agent 1 must return PatientProfile")
        profile.clinician_approved = False
        profile.reviewer = None
        return profile

    def rank_after_review(
        self,
        profile: PatientProfile,
        *,
        top_k_diseases: int = 20,
        disease_gene_associations: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        request_variant_ranking: bool = False,
        vcf_records: Sequence[Mapping[str, Any]] | None = None,
        include_source_text: bool = False,
        clinical_context: Mapping[str, Any] | None = None,
        disease_priors: Any = None,
        probability_mode: str = "none",
        require_probability: bool = False,
        require_calibrated: bool = False,
        unknown_prior: float | None = None,
        min_candidate_coverage: float = 0.0,
        abstain_when_unknown_dominates: bool = False,
    ) -> dict[str, Any]:
        require_clinician_approval(profile)
        # A calibrated posterior is still a probability.  Treat the stronger
        # request as implying the weaker one so a missing prior fails closed
        # before any downstream stage can mistake a ranking for a probability.
        effective_require_probability = require_probability or require_calibrated
        agent2_request = profile_to_agent2_request(
            profile,
            require_reviewed=True,
            clinical_context=clinical_context,
        )
        observations = agent2_request["observations"]
        agent2_result_payload: dict[str, Any] | None = None
        analyze_request = getattr(self.agent2, "analyze_agent1_request", None)
        analyze = getattr(self.agent2, "analyze", None)
        analysis_kwargs = {
            "priors": disease_priors,
            "probability_mode": probability_mode,
            "top_k": top_k_diseases,
            "require_probability": effective_require_probability,
            "require_calibrated": require_calibrated,
            "unknown_prior": unknown_prior,
            "min_candidate_coverage": min_candidate_coverage,
            "abstain_when_unknown_dominates": abstain_when_unknown_dominates,
        }
        if callable(analyze_request):
            agent2_result = analyze_request(agent2_request, **analysis_kwargs)
            disease_candidates = list(agent2_result.candidates)
            agent2_result_payload = agent2_result.as_dict()
        elif callable(analyze):
            context_values = dict(agent2_request.get("clinical_context") or {})
            allowed_context_keys = {
                "fetal_sex",
                "gestational_age_weeks",
                "maternal_age_years",
                "maternal_age_permitted",
                "family_history",
                "referral_setting",
                "cohort_id",
                "hpo_release",
            }
            agent2_context = ClinicalContext(
                case_id=profile.case_id,
                **{
                    key: context_values[key]
                    for key in allowed_context_keys
                    if key in context_values
                },
            )
            agent2_result = analyze(
                observations,
                context=agent2_context,
                **analysis_kwargs,
            )
            disease_candidates = list(agent2_result.candidates)
            agent2_result_payload = agent2_result.as_dict()
        else:
            if (
                disease_priors is not None
                or probability_mode != "none"
                or require_probability
                or require_calibrated
            ):
                raise RuntimeError(
                    "This Agent 2 implementation does not support probability-aware analyze()"
                )
            disease_candidates = self.agent2.rank(observations, top_k=top_k_diseases)

        candidate_payloads = (
            list(agent2_result_payload.get("candidates", []))
            if agent2_result_payload is not None
            else []
        )
        agent2_abstained = bool(
            agent2_result_payload is not None
            and agent2_result_payload.get("abstained") is True
        )
        calibrated_probability_count = sum(
            1
            for item in candidate_payloads
            if item.get("estimated_probability") is not None
            and item.get("calibrated") is True
            and item.get("probability_kind") == "CALIBRATED_PROBABILITY"
        )
        probability_candidate_count = sum(
            1 for item in candidate_payloads if item.get("estimated_probability") is not None
        )

        # Do not rely solely on one Agent-2 implementation to honour the flag.
        # The built-in matcher returns a structured abstention because it has no
        # posterior calibrator.  Any alternate implementation that silently
        # returns ranking scores or uncalibrated Bayes estimates is a contract
        # violation and must not reach Agent 3.
        if require_calibrated and not agent2_abstained:
            all_candidates_are_calibrated_posteriors = bool(candidate_payloads) and (
                calibrated_probability_count == len(candidate_payloads)
            )
            if not all_candidates_are_calibrated_posteriors:
                raise RuntimeError(
                    "Agent 2 violated require_calibrated: every returned candidate must "
                    "contain estimated_probability with calibrated=true and "
                    "probability_kind=CALIBRATED_PROBABILITY, or Agent 2 must abstain"
                )
        if effective_require_probability and not require_calibrated and not agent2_abstained:
            if not candidate_payloads or probability_candidate_count != len(candidate_payloads):
                raise RuntimeError(
                    "Agent 2 violated require_probability: every returned candidate must "
                    "contain estimated_probability, or Agent 2 must abstain"
                )

        agent3_result = self.agent3.rank(
            disease_candidates,
            disease_gene_associations=disease_gene_associations,
            case_id=profile.case_id,
            ontology_version=profile.ontology_version,
            request_variant_ranking=request_variant_ranking,
            vcf_records=vcf_records,
        )
        profile_payload = jsonable(profile)
        if not include_source_text:
            profile_payload["source_text"] = "[REDACTED_BY_DEFAULT]"
        agent2_input_summary = {
            "schema_version": agent2_request["schema_version"],
            "ontology_version": agent2_request["ontology_version"],
            "observations": [
                {
                    key: item.get(key)
                    for key in (
                        "hpo_id",
                        "hpo_name",
                        "status",
                        "subject",
                        "assertion_certainty",
                        "extraction_confidence",
                        "linking_confidence",
                        "reviewer_status",
                        "gestational_age_weeks",
                        "observation_method",
                        "modifiers",
                    )
                }
                for item in agent2_request["observations"]
            ],
            "contextual_observation_count": len(agent2_request["contextual_observations"]),
            "ignored_mention_count": len(agent2_request["ignored_mentions"]),
            "clinical_context": agent2_request["clinical_context"],
            "warnings": agent2_request["warnings"],
            "raw_text_included": False,
        }
        return {
            "case_id": profile.case_id,
            "research_use_only": True,
            "diagnostic_claim": False,
            "review": {"approved": True, "reviewer": profile.reviewer},
            "agent1": profile_payload,
            "agent2_input": agent2_input_summary,
            "agent2": [candidate.as_dict() for candidate in disease_candidates],
            "agent2_result": agent2_result_payload,
            "agent2_probability_semantics": {
                "compatibility_score_is_probability": False,
                "confidence_score_is_clinical_posterior": False,
                "ranking_calibrator_changes_estimated_probability": False,
                "estimated_probability_requires_prior": True,
                "uncalibrated_bayes_estimate_is_clinical_probability": False,
                "probability_mode": probability_mode,
                "probability_output_available": agent2_result_payload is not None,
                "agent2_abstained": agent2_abstained,
                "downstream_candidate_count": len(disease_candidates),
                "posterior_calibration_requested": require_calibrated,
                "posterior_calibration_request_satisfied": (
                    None
                    if not require_calibrated
                    else (
                        not agent2_abstained
                        and bool(candidate_payloads)
                        and calibrated_probability_count == len(candidate_payloads)
                    )
                ),
                "calibrated_probability_count": calibrated_probability_count,
                "probability_candidate_count": probability_candidate_count,
            },
            "agent3": agent3_result,
        }

    def analyze_prenatal_decision(
        self,
        profile: PatientProfile,
        *,
        top_k_candidates: int = 5,
        prenatal_agent: PrenatalDecisionAgent | None = None,
    ) -> dict[str, Any]:
        """Synthesize clinical pattern, rerank candidates for prenatal context, and recommend genetic tests."""
        require_clinician_approval(profile)
        observations = [
            PhenotypeObservation(
                hpo_id=m.hpo_id,
                status=m.assertion.value if hasattr(m.assertion, "value") else str(m.assertion),
            )
            for m in profile.mentions
            if m.hpo_id
        ]
        disease_candidates = self.agent2.rank(observations, top_k=20)
        decider = prenatal_agent or PrenatalDecisionAgent()
        report = decider.analyze_case(
            case_id=profile.case_id,
            observations=observations,
            disease_candidates=disease_candidates,
            top_k=top_k_candidates,
        )
        return {
            "case_id": profile.case_id,
            "clinical_pattern": report.clinical_pattern,
            "syndromes_text": report.syndromes_text,
            "genetic_tests_text": report.genetic_tests_text,
            "top_candidates": [
                {
                    "rank": c.rank,
                    "id": c.disease_id,
                    "name": c.disease_name,
                    "original_rank": c.original_rank,
                    "rerank_score": c.rerank_score,
                    "rationale": c.clinical_rationale,
                    "recommended_tests": c.recommended_tests,
                }
                for c in report.top_candidates
            ],
            "structured_payload": report.structured_payload,
        }
