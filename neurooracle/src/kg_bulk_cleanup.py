"""Exact metadata redundancy rules; no scientific values or seals rewritten."""
from copy import deepcopy
import json

EMPTY_HINTS = ('subject_canonical_hint','object_canonical_hint','subject_atlas','object_atlas')
SCOPE_ALIASES = dict(scope_confidence='confidence', scope_decision_basis='decision_basis',
    scope_rubric_version='rubric_version', scope_review_status='review_status', scope_assignment_stage='review_stage')
SCIENTIFIC_ALIASES = ('subject_type','object_type','conditions','population')


def identical(left, right):
    return type(left) is type(right) and json.dumps(left,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(right,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def legacy_scope_metadata(payload):
    """Compatibility view, never stored: existing conflicting values win."""
    md=deepcopy(payload.get('metadata') or {})
    audit=payload.get('scope_reaudit') or {}
    if not isinstance(md,dict) or not isinstance(audit,dict):raise TypeError('scope holders must be objects')
    for alias,key in SCOPE_ALIASES.items():
        if alias not in md and key in audit:md[alias]=deepcopy(audit[key])
    for key in SCIENTIFIC_ALIASES:
        if key not in md and key in payload:md[key]=deepcopy(payload[key])
    return md


def simplify_record(kind, record, changes=None):
    """Only whitelisted empty hints and byte/type-equal canonical duplicates."""
    if kind not in ('node','edge'):raise ValueError('expected node or edge')
    if kind!='node' or not str(record.get('id') or '').startswith('CLM:'):return record
    md=record.get('metadata');inner=md.get('metadata') if isinstance(md,dict) else None
    if inner is None:return record
    if not isinstance(inner,dict):raise TypeError('nested claim metadata must be an object')
    audit=md.get('scope_reaudit') or {}
    if not isinstance(audit,dict):raise TypeError('scope audit must be an object')
    drops={}
    for key in EMPTY_HINTS:
        if key in inner and type(inner[key]) is str and inner[key]=='':drops[key]='empty_hint'
    if 'raw_stats' in inner and type(inner['raw_stats']) is dict and not inner['raw_stats']:
        drops['raw_stats']='empty_raw_stats'
    for key in SCIENTIFIC_ALIASES:
        # Falsy population/conditions have legacy fallback semantics. Do not
        # infer that {}, [], False, 0, or an absent value are interchangeable.
        if key in inner and key in md and bool(md[key]) and identical(inner[key],md[key]):
            drops[key]='canonical_science_duplicate'
    for alias,key in SCOPE_ALIASES.items():
        if alias in inner and key in audit and identical(inner[alias],audit[key]):
            drops[alias]='canonical_audit_duplicate'
    if not drops:return record
    out=dict(record);out['metadata']=dict(md);out['metadata']['metadata']=dict(inner)
    for key,reason in drops.items():
        del out['metadata']['metadata'][key]
        if changes is not None:
            metric=reason+':node/claim.metadata.metadata.'+key
            changes[metric]=changes.get(metric,0)+1
    return out


def verify_simplification(original, candidate):
    """Independent rule check: retained keys/types, canonical witnesses and seals."""
    if set(original)!=set(candidate):raise ValueError('top-level shape changed')
    if not identical({k:v for k,v in original.items() if k!='metadata'},
                     {k:v for k,v in candidate.items() if k!='metadata'}):raise ValueError('node fields changed')
    old,new=original['metadata'],candidate['metadata']
    if set(old)!=set(new) or not identical({k:v for k,v in old.items() if k!='metadata'},
                                          {k:v for k,v in new.items() if k!='metadata'}):
        raise ValueError('scientific payload or audit changed')
    oi,ni=old.get('metadata'),new.get('metadata')
    if oi is None:
        if ni is not None:raise ValueError('new metadata holder')
        return
    if not set(ni)<=set(oi):raise ValueError('new nested field')
    if any(not identical(v,oi[k]) for k,v in ni.items()):raise ValueError('retained value changed')
    audit=old.get('scope_reaudit') or {}
    for key in set(oi)-set(ni):
        value=oi[key]
        empty=key in EMPTY_HINTS and type(value) is str and value==''
        empty=empty or (key=='raw_stats' and type(value) is dict and not value)
        science=key in SCIENTIFIC_ALIASES and key in old and bool(old[key]) and identical(old[key],value)
        review=key in SCOPE_ALIASES and SCOPE_ALIASES[key] in audit and identical(audit[SCOPE_ALIASES[key]],value)
        if not (empty or science or review):raise ValueError('unapproved removed value: '+key)


def symmetric_correlation_key(key):
    """Index proposal only; never reverse predicts/causes or original records."""
    key=tuple(key)
    if len(key)!=5:raise ValueError('complete five-field key required')
    if key[2]=='correlates_with' and key[:2]>key[3:]:return (*key[3:],key[2],*key[:2])
    return key
