"""Finite, loss-bounded Batch 2 repair operations. No writer, model or fuzzy merge.

Study-design normalization is a literal vocabulary contract, not source review.
The repair plan must separately bind current records, ontology proofs, edge
closure and primary-source decisions before a future candidate is built.
"""
from copy import deepcopy
import hashlib

from .kg_identity_pilot import digest
from .kg_literal_endpoint_repair import edge_owner
from .schema import ConceptNode

VERSION = 'kg.root_repairs.v1'
# These exact strings describe the curation workflow, not the original study.
# Mixed review/method labels are deliberately NOT included.
WORKFLOW_STUDY_TYPES = frozenset({
    'manual_curated_abstract', 'manual_abstract_review', 'manual_audit_existing_phase2_claim',
    'manual_abstract_curation', 'manual_case2_abstract_curation',
    'manual neuroimaging abstract curation', 'manual_new_literature_abstract',
})
DESIGN_ALIASES = {
    'case report': 'case_report', 'case-control': 'case_control', 'case control': 'case_control',
    'case-control study': 'case_control', 'case series': 'case_series',
    'cross-sectional': 'cross_sectional', 'cross-sectional study': 'cross_sectional',
    'cross sectional': 'cross_sectional', 'cross sectional study': 'cross_sectional',
    'observational study': 'observational', 'observational_study': 'observational',
    'cohort study': 'cohort', 'longitudinal study': 'longitudinal',
    'longitudinal cohort': 'longitudinal_cohort', 'retrospective cohort': 'retrospective_cohort',
    'prospective cohort': 'prospective_cohort',
    'randomized controlled trial': 'randomized_controlled_trial',
    'randomised controlled trial': 'randomized_controlled_trial',
    'randomised_controlled_trial': 'randomized_controlled_trial',
    'clinical trial': 'clinical_trial', 'narrative review': 'narrative_review',
    'clinical review': 'clinical_review', 'systematic review': 'systematic_review',
    'meta-analysis': 'meta_analysis', 'meta analysis': 'meta_analysis',
    'systematic review and meta-analysis': 'systematic_review_and_meta_analysis',
}
METHOD_STUDY_TYPES = frozenset({'fMRI', 'EEG', 'sMRI', 'PET', 'DTI', 'resting-state fMRI', 'structural MRI', 'GWAS'})


def metadata_repair(record):
    """Keep missing/unknown/mixed values; never guess a design from a method."""
    md = record.get('metadata') or {}
    evidence = md.get('evidence')
    if not isinstance(evidence, dict) or not isinstance(evidence.get('study_type'), str):
        return record, None
    value = evidence['study_type']
    reason = ('workflow_label_removed_from_science' if value in WORKFLOW_STUDY_TYPES else
              'exact_design_alias' if value in DESIGN_ALIASES else
              'method_moved_losslessly' if value in METHOD_STUDY_TYPES else None)
    if not reason: return record, None
    if reason == 'method_moved_losslessly' and not isinstance(evidence.get('methodology', ''), (str, type(None))):
        return record, 'hold_nonstring_methodology'
    out = deepcopy(record); e = out['metadata']['evidence']
    if reason == 'exact_design_alias':
        e['study_type'] = DESIGN_ALIASES[value]
    else:
        del e['study_type']
        if reason == 'method_moved_losslessly':
            existing = e.get('methodology')
            # Exact segments only: do not infer that MRI implies fMRI, or that
            # a substring in a compound description is redundant evidence.
            if not existing:
                e['methodology'] = value
            elif value not in existing.split('; '):
                e['methodology'] = existing + '; ' + value
    return out, reason


def changes(before, after, path=()):
    if digest(before) == digest(after): return []
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            p = (*path, key)
            if key not in before or key not in after:
                result.append(dict(path=list(p), old_exists=key in before, new_exists=key in after,
                    **({'old': deepcopy(before[key])} if key in before else {}),
                    **({'new': deepcopy(after[key])} if key in after else {})))
            else: result.extend(changes(before[key], after[key], p))
        return result
    return [dict(path=list(path), old_exists=True, new_exists=True, old=deepcopy(before), new=deepcopy(after))]


def apply_fields(record, fields):
    out = deepcopy(record); seen = []
    for change in fields:
        path = tuple(change['path'])
        if not path or any(path[:len(p)] == p or p[:len(path)] == path for p in seen):
            raise ValueError('overlapping or empty field change')
        seen.append(path); holder = out
        for key in path[:-1]:
            if not isinstance(holder, dict) or key not in holder: raise ValueError('missing field parent')
            holder = holder[key]
        key = path[-1]
        if not isinstance(holder, dict) or (key in holder) != change['old_exists']:
            raise ValueError('field presence changed')
        if change['old_exists'] and digest(holder[key]) != digest(change['old']):
            raise ValueError('field value changed')
        if change['new_exists']: holder[key] = deepcopy(change['new'])
        else: del holder[key]
    return out


def event(before, after, **details):
    fields = changes(before, after)
    if not fields or digest(apply_fields(before, fields)) != digest(after):
        raise ValueError('invalid or empty finite change')
    return dict(record_sha256=digest(before), proposed_sha256=digest(after), field_changes=fields, **details)


def apply_event(record, proposal):
    if digest(record) != proposal['record_sha256']: raise ValueError('stale record')
    out = apply_fields(record, proposal['field_changes'])
    if digest(out) != proposal['proposed_sha256']: raise ValueError('unexpected proposed record')
    return out


def endpoint_change(record, side, target):
    if side not in {'subject', 'object'} or target.startswith('CLM:'):
        raise ValueError('invalid entity endpoint')
    out = deepcopy(record); md = out['metadata']; field = side + '_id'; old = md[field]
    inner = md.get('metadata') or {}
    if field in inner:
        if inner[field] != old: raise ValueError('conflicting nested endpoint')
        inner[field] = target
    md[field] = target
    if md['subject_id'] == md['object_id']: raise ValueError('new self relation')
    return out


def owned_endpoint_changes(original, revised, owned):
    before, after = original['metadata'], revised['metadata']; cid = original['id']
    events = []; seen = set()
    for ordinal, edge in owned:
        if edge_owner(edge) != cid: raise ValueError('edge owner differs')
        out = deepcopy(edge)
        if edge['relation_type'] == 'about':
            sides = [s for s in ('subject', 'object') if before[s + '_id'] == edge['target_id']]
            if len(sides) != 1: raise ValueError('ambiguous about endpoint')
            side = sides[0]
            if side in seen: raise ValueError('duplicate about edge')
            seen.add(side); out['target_id'] = after[side + '_id']
        else:
            if (edge['source_id'], edge['target_id'], edge['relation_type']) != (
                    before['subject_id'], before['object_id'], before['predicate']):
                raise ValueError('science edge disagrees with claim')
            out['source_id'], out['target_id'], out['relation_type'] = after['subject_id'], after['object_id'], after['predicate']
        if edge != out:
            if out['source_id'] == out['target_id']: raise ValueError('new self loop')
            events.append(event(edge, out, ordinal=ordinal, claim_id=cid))
    # Shared-relation dedup may already omit a science edge. It must not be
    # invented here; the two claim-to-entity about edges still prove closure.
    if seen != {'subject', 'object'}: raise ValueError('incomplete about closure')
    return events


def unclassified_literal(name):
    if not isinstance(name, str) or not name.strip(): raise ValueError('empty full mention')
    # This represents the original complete text, not a guessed ontology type.
    nid = 'CLM_CONCEPT:unclassified_mention_' + hashlib.sha256((VERSION + '|' + name).encode()).hexdigest()
    return ConceptNode(id=nid, preferred_name=name, domain_tags=['external'], source_vocab='claim_extraction').to_dict()


def duplicate_node_payload(record):
    out = deepcopy(record); out.pop('id')
    out['metadata'].pop('id', None); out['metadata'].pop('scope_reaudit', None)
    # Confidence, complete source metadata, negative/context evidence and
    # unknown fields intentionally remain, unlike the broader candidate key.
    return out


def audit_decision(record):
    audit = record['metadata'].get('scope_reaudit') or {}
    ignored = {'reviewed_at', 'review_stage', 'reviewer_id', 'reasoning_effort', 'decision_basis',
               'claim_evidence_sha256', 'audit_key'}
    return {k: v for k, v in audit.items() if k not in ignored}


def normalized_owner(edge, cid):
    """Normalize exact owner references only; never rewrite other relation ends."""
    out = deepcopy(edge)
    if edge_owner(out) != cid: raise ValueError('foreign edge')
    if out['relation_type'] == 'about': out['source_id'] = 'CLM:owner'
    md = out.get('metadata') or {}
    if md.get('claim_id') == cid: md['claim_id'] = 'CLM:owner'
    if out.get('source') == 'claim:' + cid: out['source'] = 'claim:CLM:owner'
    return out
