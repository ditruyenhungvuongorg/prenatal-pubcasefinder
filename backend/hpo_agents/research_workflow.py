"""Explicit unreviewed research runs, without impersonating clinician approval."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from typing import Any, Mapping

from .agent1_contract import profile_to_agent2_request
from .agent2_schema import ClinicalContext
from .hpo_catalog import HPOCatalog
from .schemas import PatientProfile, jsonable


def propose_hpo(
    profile: PatientProfile,
    catalog: HPOCatalog,
    *,
    selections: Mapping[int, str | None] | None = None,
    top_k: int = 5,
    min_suggestion_score: float = 0.55,
) -> PatientProfile:
    """Return a new unreviewed profile with at most one provisional ID per mention.

    Only an unambiguous exact match to the original mention is auto-selected.
    Approximate matches on the explicit lexicon are suggestions, not semantic
    equivalence or probabilities. A caller may choose one of the shown IDs or
    explicitly leave the mention unresolved with None.
    """
    if not isinstance(profile, PatientProfile):
        raise TypeError("profile must be a PatientProfile")
    if not 1 <= top_k <= 5:
        raise ValueError("top_k must be between 1 and 5")
    if not 0 <= min_suggestion_score <= 1:
        raise ValueError("min_suggestion_score must be between 0 and 1")
    selections = dict(selections or {})
    if any(type(i) is not int or i < 0 or i >= len(profile.mentions) for i in selections):
        raise ValueError("selections must use valid mention indices")
    result = deepcopy(profile)
    result.clinician_approved = False
    result.reviewer = None
    result.ontology_version = catalog.version
    result.provenance = {"workflow": "research_unreviewed", "upstream": deepcopy(profile.provenance)}
    for i, mention in enumerate(result.mentions):
        if profile.source_text and profile.source_text[mention.span_start:mention.span_end] != mention.mention_text:
            raise ValueError(f"mention {i} does not match its source span")
        candidates = catalog.retrieve(mention.mention_text, top_k=max(2, top_k), allow_fuzzy=True)
        candidates = [c for c in candidates if c.score >= min_suggestion_score and catalog.validate(c.hpo_id)]
        exact = [c for c in candidates if c.retrieval_method == "exact_lexicon"]
        mention.candidates = candidates[:top_k]
        automatic = exact[0] if len(exact) == 1 else None
        selected = automatic
        basis = "unique_exact_provisional" if selected else "unresolved"
        if i in selections:
            wanted = selections[i]
            selected = next((c for c in mention.candidates if c.hpo_id == wanted), None)
            if wanted is not None and selected is None:
                raise ValueError(f"selection for mention {i} is outside the displayed candidates")
            basis = "user_selected_unreviewed" if selected else "user_left_unresolved"
        mention.hpo_id = selected.hpo_id if selected else None
        mention.hpo_name = selected.hpo_name if selected else None
        mention.linking_confidence = selected.score if selected else None
        mention.model_confidence = mention.linking_confidence
        mention.needs_review = True
        mention.provenance = {
            "reviewer_status": "unreviewed",
            "selection_basis": basis,
            "retrieval_score_is_probability": False,
            "upstream": deepcopy(profile.mentions[i].provenance),
        }
    return result


def rank_research(
    pipeline: Any,
    profile: PatientProfile,
    catalog: HPOCatalog,
    *,
    selections: Mapping[int, str | None] | None = None,
    top_k_diseases: int = 20,
    include_source_text: bool = False,
    disease_gene_associations: Any = None,
) -> dict[str, Any]:
    """Rank one explicit provisional interpretation with the research score API.

    This separate opt-in entry point never calls or weakens the reviewed
    handoff parser, never marks data approved and never writes training labels.
    """
    if top_k_diseases < 1:
        raise ValueError("top_k_diseases must be positive")
    proposed = propose_hpo(profile, catalog, selections=selections)
    request = profile_to_agent2_request(proposed, require_reviewed=False)
    observations = request["observations"]
    positive = any(o['status'] in {'present', 'suspected'} for o in observations)
    # A contradictory pair must be resolved before ranking, even in research.
    states: dict[str, set[str]] = {}
    for o in observations:
        states.setdefault(o['hpo_id'], set()).add(o['status'])
    conflict = any('absent' in s and bool(s & {'present', 'suspected'}) for s in states.values())
    reason = 'conflicting_provisional_observations' if conflict else None if positive else 'no_positive_fetal_hpo'
    candidates = []
    agent2_payload = None
    if reason is None:
        analyze = getattr(pipeline.agent2, 'analyze', None)
        if callable(analyze):
            allowed = {f.name for f in fields(ClinicalContext)} - {'case_id'}
            context = ClinicalContext(case_id=profile.case_id, **{
                k: v for k, v in request['clinical_context'].items() if k in allowed
            })
            result = analyze(observations, context=context, probability_mode='none', top_k=top_k_diseases)
            agent2_payload = result.as_dict()
            if agent2_payload.get('abstained'):
                reason = 'agent2_abstained'
            else:
                candidates = list(result.candidates)
        else:
            candidates = list(pipeline.agent2.rank(observations, top_k=top_k_diseases))
    agent3_payload = None if reason else pipeline.agent3.rank(
        candidates, disease_gene_associations=disease_gene_associations,
        case_id=profile.case_id, ontology_version=proposed.ontology_version,
    )
    payload = jsonable(proposed)
    if not include_source_text:
        payload['source_text'] = '[REDACTED_BY_DEFAULT]'
    suggestions = [[{
        **jsonable(c), 'definition': catalog.terms[c.hpo_id].definition,
        'selection_basis': m.provenance['selection_basis'],
        'score_is_probability': False,
    } for c in m.candidates] for m in proposed.mentions]
    return {
        'case_id': profile.case_id, 'mode': 'research_unreviewed',
        'research_use_only': True, 'diagnostic_claim': False, 'training_eligible': False,
        'review': {'approved': False, 'reviewer': None}, 'agent1': payload,
        'hpo_suggestions': suggestions, 'agent2_input': request,
        'agent2': [c.as_dict() for c in candidates], 'agent2_result': agent2_payload,
        'agent3': agent3_payload, 'abstained': reason is not None, 'abstention_reason': reason,
        'probability_mode': 'none', 'warnings': request['warnings'] + [
            'Provisional HPO interpretation; not clinician reviewed.',
            'Only selected HPO IDs were ranked; alternatives were not combined.',
        ],
    }
