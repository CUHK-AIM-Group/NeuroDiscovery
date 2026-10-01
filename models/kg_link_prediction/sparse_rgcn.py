"""Exact RGCNConv mean aggregation, materializing only active rows per relation.

PyG's dense per-relation intermediate has N rows even for a rare relation.
The computation here uses the same weights/normalization and omits only zeros.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def relation_blocks(triples, n_entities, n_relations, device):
    blocks = []
    for relation in range(n_relations):
        edges = triples[triples[:, 1] == relation]
        for reverse in (False, True):
            if not len(edges):
                continue
            source, target = (edges[:, 2], edges[:, 0]) if reverse else (edges[:, 0], edges[:, 2])
            destinations, row, counts = np.unique(target, return_inverse=True, return_counts=True)
            matrix = torch.sparse_coo_tensor(
                torch.as_tensor(np.stack((row, source)), dtype=torch.long, device=device),
                torch.as_tensor(1.0/counts[row], dtype=torch.float32, device=device),
                (len(destinations), n_entities), device=device,
            ).coalesce()
            blocks.append((relation + (n_relations if reverse else 0),
                           torch.as_tensor(destinations, dtype=torch.long, device=device), matrix))
    return blocks


def sparse_layer(layer, x, blocks):
    output = x @ layer.root if layer.root is not None else torch.zeros_like(x)
    for relation, destinations, matrix in blocks:
        aggregated = torch.sparse.mm(matrix, x)
        messages = aggregated @ layer.weight[relation]
        output.index_add_(0, destinations, messages)
    if layer.bias is not None:
        output = output + layer.bias
    return output


def encode_sparse(model, blocks, *, checkpoint_layers=False):
    if model.model_name != 'rgcn':
        raise ValueError('This exact implementation is for R-GCN only')
    x = model.entity_embedding.weight
    for layer in model.layers:
        def forward(value, current=layer):
            return model.dropout(torch.relu(sparse_layer(current, value, blocks)))
        x = checkpoint(forward, x, use_reentrant=False) if checkpoint_layers else forward(x)
    return x


def triple_keys(triples, entities, relations):
    rows = np.asarray(triples, dtype=np.uint64)
    return (rows[:, 0] * np.uint64(relations) + rows[:, 1]) * np.uint64(entities) + rows[:, 2]


def sample_typed_negatives(positive, domains, trained_nodes, known_keys, n_relations, rng, *, allow_missing=False):
    """One type-matched corruption per positive, filtered against recorded edges."""
    positive = np.asarray(positive, dtype=np.int64)
    result = positive.copy()
    remaining = np.arange(len(positive))
    pools = {int(d): np.flatnonzero((domains == d) & trained_nodes) for d in np.unique(domains)}
    for attempt in range(100):
        if not len(remaining):
            return (result,np.ones(len(result),dtype=bool)) if allow_missing else result
        result[remaining] = positive[remaining]
        head = rng.random(len(remaining)) < .5
        for column, selection in ((0, head), (2, ~head)):
            positions = remaining[selection]
            group = domains[positive[positions, column]]
            for domain in np.unique(group):
                ids = positions[group == domain]
                choices = pools[int(domain)]
                if len(choices):
                    result[ids, column] = choices[rng.integers(len(choices), size=len(ids))]
        keys = triple_keys(result[remaining], len(domains), n_relations)
        locations = np.searchsorted(known_keys, keys)
        known = (locations < len(known_keys)) & (known_keys[np.minimum(locations, len(known_keys)-1)] == keys)
        invalid = known | (result[remaining, 0] == result[remaining, 2])
        remaining = remaining[invalid]
    # A few rare type pairs are saturated. Exhaustively check those rows rather
    # than dropping their positives, inventing false relations, or widening types.
    valid = np.ones(len(result),dtype=bool)
    for position in remaining:
        valid[position] = False
        for column in (0,2):
            choices = pools[int(domains[positive[position,column]])]
            trial = np.repeat(positive[position:position+1],len(choices),axis=0)
            trial[:,column] = choices
            keys = triple_keys(trial,len(domains),n_relations)
            locations = np.searchsorted(known_keys,keys)
            observed = (locations<len(known_keys)) & (known_keys[np.minimum(locations,len(known_keys)-1)]==keys)
            permitted = np.flatnonzero(~observed & (trial[:,0]!=trial[:,2]))
            if len(permitted):
                result[position] = trial[permitted[0]]
                valid[position] = True
                break
    if allow_missing:
        return result,valid
    if not valid.all():
        raise ValueError(f'No type-matched unobserved contrast exists for {int((~valid).sum())} rows')
    return result
