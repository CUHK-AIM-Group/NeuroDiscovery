"""Apply the finite R69 contract; never broaden its scientific decisions.

Only compact field changes and digests are retained. Source records are not
saved. The independent reader reverses every retained change and compares the
ordered canonical digests with the source pass, including unaffected records.
"""
from copy import deepcopy
import hashlib

from .kg_identity_pilot import digest
from .kg_root_repairs import apply_event


def unique_index(records, key):
    result = {}
    for row in records:
        value = row[key]
        if value in result:
            raise ValueError('duplicate operation identity')
        result[value] = row
    return result


def reverse_event(record, proposal):
    """Independent inverse, including absent fields and unknown siblings."""
    if digest(record) != proposal['proposed_sha256']:
        raise ValueError('candidate record differs from approved event')
    out = deepcopy(record)
    paths = []
    for change in proposal['field_changes']:
        path = tuple(change['path'])
        if not path or any(path[:len(p)] == p or p[:len(path)] == path for p in paths):
            raise ValueError('overlapping or empty change')
        paths.append(path)
        holder = out
        for key in path[:-1]:
            if not isinstance(holder, dict) or key not in holder:
                raise ValueError('candidate field parent missing')
            holder = holder[key]
        key = path[-1]
        if not isinstance(holder, dict) or (key in holder) != change['new_exists']:
            raise ValueError('candidate field presence differs')
        if change['new_exists'] and digest(holder[key]) != digest(change['new']):
            raise ValueError('candidate field differs')
        if change['old_exists']:
            holder[key] = deepcopy(change['old'])
        else:
            del holder[key]
    if digest(out) != proposal['record_sha256']:
        raise ValueError('inverse source digest differs')
    return out


class RootTransform:
    def __init__(self, claims, edges, deleted_claims, deleted_edges, new_nodes):
        self.claims = unique_index(claims, 'claim_id')
        self.edges = unique_index(edges, 'ordinal')
        self.deleted_claims = unique_index(deleted_claims, 'claim_id')
        self.deleted_edges = unique_index(deleted_edges, 'ordinal')
        self.new_nodes = unique_index(new_nodes, 'id')
        if set(self.claims) & set(self.deleted_claims) or set(self.edges) & set(self.deleted_edges):
            raise ValueError('edit and removal overlap')
        if any(not k.startswith('CLM:') for k in self.claims.keys() | self.deleted_claims.keys()):
            raise ValueError('claim operation targets a non-claim')
        if any(not k.startswith('CLM_CONCEPT:unclassified_mention_') for k in self.new_nodes):
            raise ValueError('unexpected added entity kind')
        self.seen = {k: set() for k in ('claims', 'edges', 'deleted_claims', 'deleted_edges')}
        self.source_digests = {k: hashlib.sha256() for k in ('nodes', 'edges')}

    def records(self, records):
        added = False
        current_edge = 0
        for kind, key, original in records:
            if kind == 'metadata':
                yield kind, key, original
                continue
            if kind == 'edge' and not added:
                for nid, row in self.new_nodes.items():
                    yield 'node', nid, row
                added = True
            if kind == 'node' and key in self.new_nodes:
                raise ValueError('new node collides with source')
            operation_key = key if kind == 'node' else int(key)
            deletes = self.deleted_claims if kind == 'node' else self.deleted_edges
            edits = self.claims if kind == 'node' else self.edges
            label = 'claims' if kind == 'node' else 'edges'
            if operation_key in deletes:
                if digest(original) != deletes[operation_key]['record_sha256']:
                    raise ValueError('source deletion digest differs')
                self.seen['deleted_' + label].add(operation_key)
                continue
            self.source_digests[kind + 's'].update(bytes.fromhex(digest(original)))
            out = original
            if operation_key in edits:
                out = apply_event(original, edits[operation_key])
                self.seen[label].add(operation_key)
            if kind == 'edge':
                current_edge += 1
                key = current_edge
            yield kind, key, out
        if not added:
            for nid, row in self.new_nodes.items():
                yield 'node', nid, row
        for name, seen in self.seen.items():
            if seen != set(getattr(self, name)):
                raise ValueError('incomplete operation closure: ' + name)


def source_edge_ordinal(current, removed):
    source = current
    for ordinal in sorted(removed):
        if ordinal <= source:
            source += 1
        else:
            break
    return source
