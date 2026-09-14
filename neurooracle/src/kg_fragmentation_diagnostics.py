"""Read-only diagnostic probes. None of these relaxed keys authorizes a merge.

The accepted runtime identity is unchanged. Missing scope is unknown, not
equal scope. Counterfactual grouping measures search opportunities, not recall
against a semantic gold standard and not literature novelty.
"""
from collections import Counter
import re
import unicodedata

from .kg_identity_pilot import digest
from .relation_evidence import name_key

VERSION = 'kg.fragmentation_diagnostics.v1'
PROBES = ('rid', 'surface', 'folded', 'physical', 'orthographic', 'registry',
          'alias_route', 'predicate_family', 'orientation', 'scope_probe', 'topic')
FAMILIES = {
    'is_associated_with': 'association', 'correlates_with': 'association',
    'causes': 'causal', 'increases': 'causal_increase', 'reduces': 'causal_reduce',
    'predicts': 'prediction', 'distinguishes': 'discrimination',
}
LATERALITY = ('left', 'right', 'bilateral')
AGGREGATION = ('mean', 'sum', 'total', 'average')
DIRECTION = ('increased', 'decreased', 'reduced', 'smaller', 'larger', 'higher', 'lower')
MEASURES = ('volume', 'thickness', 'surface area', 'fractional anisotropy',
            'mean diffusivity', 'radial diffusivity', 'axial diffusivity',
            'functional connectivity', 'structural connectivity', 'activation',
            'atrophy', 'binding potential', 'perfusion')
MODALITIES = ('fmri', 'mri', 'pet', 'dti', 'eeg', 'meg', 'spect')
PROBE_ONLY = 'candidate_generation_only_not_scientific_equivalence'


def orthographic(value):
    value = name_key(value).casefold()
    value = value.translate(str.maketrans({c: '-' for c in '‐‑‒–—−'}))
    value = value.replace('’', "'")
    value = re.sub(r'\bgrey\b', 'gray', value)
    return ' '.join(re.sub(r'[-/]', ' ', value).split())


def features(value):
    text = orthographic(value)
    words = set(re.findall(r'\b\w+\b', text))
    def phrases(values):
        return [v for v in values if re.search(r'(?<!\w)' + re.escape(v) + r'(?!\w)', text)]
    return dict(laterality=sorted(words & set(LATERALITY)),
        aggregation=sorted(words & set(AGGREGATION)),
        direction=sorted(words & set(DIRECTION)), measures=phrases(MEASURES),
        modalities=phrases(MODALITIES), token_count=len(text.split()),
        compound=bool(re.search(r'\b(?:and|or)\b|[+&;,]', text)),
        contextual_wording=bool(re.search(r'\b(?:patients?|participants?|cohorts?|during|after|before|versus|compared)\b', text)),
        symbol_like=bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9+_.-]{1,11}', name_key(value))
                         and any(c.isupper() for c in name_key(value))),
        subtype_wording=bool(re.search(r'\b(?:risk|susceptibility|severity|subfield|expression|protein|allele|variant|gene)\b', text)))


def scoped_probe(value):
    # Deliberately unsafe probe to COUNT scope barriers; never used by runtime.
    text = orthographic(value)
    text = re.sub(r'\b(?:' + '|'.join((*LATERALITY, *AGGREGATION, *DIRECTION)) + r')\b', ' ', text)
    return ' '.join(text.split()) or orthographic(value)


def key(left, predicate, right, *, symmetric=False):
    if symmetric and left > right:
        left, right = right, left
    return digest((left, predicate, right))


def extra_keys(md, terms):
    sn, tn = orthographic(md['subject_name']), orthographic(md['object_name'])
    p = md['predicate']
    current = list(terms.relation_key(md))
    routed = []
    for side, offset in (('subject', 0), ('object', 3)):
        term = terms.term_for(md, side)
        if term and term.get('canonicalize', True):
            current[offset:offset+2] = [term['target_id'], term['canonical_name']]
            routed.append(side)
    from .kg_bulk_cleanup import symmetric_correlation_key
    from .relation_evidence import relation_id
    family = FAMILIES.get(p, p)
    return dict(orthographic=key(sn,p,tn,symmetric=p=='correlates_with'),
        registry=relation_id(symmetric_correlation_key(current)),
        predicate_family=key(sn,family,tn,symmetric=family=='association'),
        orientation=key(sn,p,tn,symmetric=p in {'correlates_with','is_associated_with'}),
        scope_probe=key(scoped_probe(md['subject_name']),family,scoped_probe(md['object_name']),symmetric=family=='association'),
        topic=key(sn,'topic_not_assertion',tn,symmetric=True)), routed


def explicit_barriers(left, right):
    """Observable differences, not a diagnosis that either source is wrong."""
    out = []
    for side in ('s','t'):
        a, b = features(left[side+'n']), features(right[side+'n'])
        for field in ('laterality','aggregation','direction','measures','modalities'):
            if a[field] and b[field] and a[field] != b[field]:
                out.append(side + '_' + field + '_different')
            elif bool(a[field]) != bool(b[field]):
                out.append(side + '_' + field + '_specified_vs_unknown')
        if a['symbol_like'] or b['symbol_like']:
            if name_key(left[side+'n']) != name_key(right[side+'n']):
                out.append(side + '_symbol_or_abbreviation_requires_context')
        if a['compound'] != b['compound']:
            out.append(side + '_compound_scope_requires_review')
    if left['p'] != right['p']:
        out.append('different_predicates_not_automatically_equivalent')
    if left['neg'] != right['neg']:
        out.append('different_negation_preserve_evidence')
    if left['ctx'] != right['ctx']:
        out.append('different_evidence_context_not_by_itself_identity_error')
    return out


def semantic_decision(left, right, *, identity_proof=False, predicate_proof=False,
                      scope_reviewed=False):
    """Executable proposed policy on review cards, not a KG mutation API.

    Return separate decisions for shared relation identity and evidence. Even
    an approved relation identity never authorizes deleting evidence members.
    """
    barriers = explicit_barriers(left,right)
    hard = [b for b in barriers if b.endswith('_different')]
    unknown = [b for b in barriers if 'specified_vs_unknown' in b or 'requires_' in b]
    if hard:
        decision = 'keep_distinct_scientific_scope'
    elif not identity_proof or unknown and not scope_reviewed:
        decision = 'hold_identity_or_scope_evidence'
    elif left['p'] != right['p'] and (not predicate_proof or
            FAMILIES.get(left['p'],left['p']) != FAMILIES.get(right['p'],right['p'])):
        decision = 'keep_distinct_predicates'
    else:
        decision = 'share_relation_preserve_each_observation'
    return dict(decision=decision,barriers=barriers,evidence_merge_allowed=False,
        independent_studies_inferred=False,scientific_consensus_inferred=False)


def partition_summary(rows):
    """Exact disjoint masks; overlapping probe counts must never be summed."""
    result = Counter()
    for row in rows:
        result[int(row['mask'])] += 1
    return [{'mask': k, 'fine_relations': v} for k,v in sorted(result.items())]
