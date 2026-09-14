"""Complete-name endpoint repairs without broadening scientific equivalence."""
from copy import deepcopy

from .kg_gene_boundary_repair import endpoint_gate as gene_gate
from .kg_identity_pilot import digest
from .kg_literal_endpoint_repair import reviewed_edges, reuse_gate, edge_owner
from .kg_root_repairs import endpoint_change, event

UNCLASSIFIED_REASONS = frozenset({
    'structural_endpoint_repaired_but_scientific_type_unresolved',
    'structural_reference_fixed_scientific_entity_type_unresolved',
})


def consolidate_root_holds(rows):
    """Join the two spellings of one issue, preserving all distinct facts."""
    merged = {}; output = []
    for source in rows:
        row = deepcopy(source)
        if row['reason'] not in UNCLASSIFIED_REASONS:
            output.append(row)
            continue
        key = (row['claim_id'], row['side'])
        reason = row.pop('reason')
        if key not in merged:
            merged[key] = dict(row, reason='scientific_entity_type_unresolved', historical_queue_reasons=[reason])
        else:
            target = merged[key]
            if reason in target['historical_queue_reasons']:
                raise ValueError('unexpected repeated issue form')
            for field, value in row.items():
                if field in target and digest(target[field]) != digest(value):
                    raise ValueError('conflicting issue facts: ' + field)
                target[field] = value
            target['historical_queue_reasons'].append(reason)
    output.extend(merged.values())
    return output


def verified_route(md, side, terms):
    """The registry flag controls label normalization, not term proof validity.

    A complete eligible mention can use its proven ID while retaining the
    disabled canonical label rule. Whole-group closure is an additional gate.
    """
    term = terms.term_for(md, side)
    if term is None or md.get(side + '_id') == term['target_id']:
        return None
    return term


def gene_literal_gate(md, side, gene_witness, target, incidents):
    reason = gene_gate(md, side, gene_witness)
    if reason:
        return reason
    if not incidents:
        return 'existing_literal_has_no_current_incidents'
    return reuse_gate(target, md[side + '_name'], incidents)


def finite_endpoint_events(record, targets, owned):
    """Require exactly two about edges and zero/one matching science edge."""
    if not targets:
        raise ValueError('no endpoint changes')
    out = record
    for side, target in sorted(targets.items()):
        out = endpoint_change(out, side, target)
    old_events = reviewed_edges(record['id'], record, out, owned)
    old_by_ordinal = dict(owned)
    edge_events = []
    for change in old_events:
        before = old_by_ordinal[change['ordinal']]
        after = deepcopy(before)
        for field, values in change['changes'].items():
            after[field] = values['new']
        edge_events.append(event(before, after, ordinal=change['ordinal'], claim_id=record['id']))
    return out, event(record, out, claim_id=record['id']), edge_events


def repair_owned_closure(record, owned):
    """Repair only proven stale original-ID references and exact edge copies.

    The scientific claim is authoritative for its own materialized edges, not
    a new source adjudication. Unfamiliar IDs, owner/polarity differences or
    nonidentical duplicate payloads are never discarded.
    """
    md = record['metadata']; cid = record['id']
    current = {s: md[s + '_id'] for s in ('subject', 'object')}
    if current['subject'] == current['object']:
        raise ValueError('self relation requires separate review')
    originals = {s: md.get('original_' + s + '_id') for s in current}
    about = [(i, e) for i, e in owned if e['relation_type'] == 'about']
    missing = {s for s in current if not any(e['target_id'] == current[s] for _, e in about)}
    invalid_about = [i for i, e in about if e['target_id'] not in current.values()]
    normalized = []
    for ordinal, row in owned:
        if edge_owner(row) != cid or (row.get('metadata') or {}).get('claim_id') not in (None, '', cid):
            raise ValueError('inconsistent owned-edge identity')
        out = deepcopy(row); em = out.get('metadata') or {}
        if 'negated' in em and (type(em['negated']) is not bool or em['negated'] is not md['negated']):
            raise ValueError('edge polarity disagrees with observation')
        if row['relation_type'] == 'about':
            if row['target_id'] in current.values():
                side = next(s for s in current if row['target_id'] == current[s])
                if em.get('anchor_role', side) != side:
                    raise ValueError('current anchor has conflicting role')
            else:
                possible = {s for s in current if originals[s] and row['target_id'] == originals[s]}
                if len(possible) == 1:
                    side = possible.pop()
                    if em.get('anchor_role', side) != side:
                        raise ValueError('historical anchor has conflicting role')
                elif len(possible) == 2 and len(missing) == 1 and len(invalid_about) == 1:
                    side = next(iter(missing))
                    if em.get('anchor_role') not in current:
                        raise ValueError('ambiguous placeholder anchor lacks explicit role')
                    out['metadata']['anchor_role'] = side
                else:
                    raise ValueError('unproved old about endpoint')
                out['target_id'] = current[side]
        else:
            if row['relation_type'] != md['predicate']:
                raise ValueError('owned predicate differs')
            for side, field in (('subject', 'source_id'), ('object', 'target_id')):
                if row[field] == current[side]:
                    continue
                if not originals[side] or row[field] != originals[side]:
                    raise ValueError('unproved old science endpoint')
                out[field] = current[side]
        normalized.append((ordinal, row, out))
    # Prefer a copy already identical to the repaired payload; unknown fields
    # and analysis-specific metadata participate in this exact comparison.
    groups = {}
    for item in normalized:
        groups.setdefault(digest(item[2]), []).append(item)
    retained = []; deletions = []
    for sig, group in groups.items():
        keeper = min(group, key=lambda x: (digest(x[1]) != sig, x[0]))
        retained.append((keeper[0], keeper[2]))
        for ordinal, original, out in group:
            if ordinal != keeper[0]:
                deletions.append(dict(ordinal=ordinal, record_sha256=digest(original), claim_id=cid,
                    kind='exact_current_owned_edge_duplicate', keeper_ordinal=keeper[0],
                    normalized_payload_sha256=sig, full_unknown_payload_preserved=True))
    retained.sort()
    # No identities are changed by this call: it is also a strict 2-about,
    # 0/1-science closure assertion for the repaired representation.
    reviewed_edges(cid, record, record, retained)
    return retained, deletions
