"""Serve a trained GNN link predictor behind the `Scorer`/`LinkScorer` surface.

`neurooracle.src.kge.hypothesis_path` scores a candidate by calling
`score_batch(triples)` on any object exposing `ent2idx` / `rel2idx`. `ComplExScorer`
already does; this adapter does the same for a message-passing GNN checkpoint
written by `models.kg_link_prediction.train`.

Message passing needs the whole adjacency, so the encoder is run once over the
supplied graph and the resulting node embeddings are cached; per-candidate
scoring is then a cheap batched decoder call. This is a scoring surface over
existing read-only graph data. It does not assert truth, novelty, or first-report
status, and it does not train or retune anything.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math

import numpy as np

import torch

from models.kg_link_prediction.gnn import (
    GNNLinkPredictor,
    directed_message_graph,
    message_graph,
)


class GNNEdgeScorer:
    """`score_batch`/`ent2idx`/`rel2idx` view of a trained GNN checkpoint."""

    def __init__(
        self,
        model: GNNLinkPredictor,
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
        graph_triples: list[tuple[str, str, str]],
        *,
        device: str = "cpu",
        checkpoint_name: str = "gnn",
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.entity_to_id = dict(entity_to_id)
        self.relation_to_id = dict(relation_to_id)
        self.checkpoint_name = checkpoint_name
        self._embeddings: torch.Tensor | None = None
        self._graph_key: tuple | None = None
        if graph_triples:
            self.refresh(graph_triples)

    # `hypothesis_path` reads these two mappings before calling into the model.
    @property
    def ent2idx(self) -> dict[str, int]:
        return self.entity_to_id

    @property
    def rel2idx(self) -> dict[str, int]:
        return self.relation_to_id

    @property
    def name(self) -> str:
        return f"{self.checkpoint_name}:{self.model.model_name}:{self.model.decoder_name}"

    def _encode_graph(self, graph_triples: list[tuple[str, str, str]]):
        encoded = []
        for source, relation, target in graph_triples:
            source_index = self.entity_to_id.get(source)
            relation_index = self.relation_to_id.get(relation)
            target_index = self.entity_to_id.get(target)
            if source_index is None or relation_index is None or target_index is None:
                raise ValueError("Graph contains an out-of-vocabulary triple; no silent edge dropping")
            encoded.append([source_index, relation_index, target_index])
        if not encoded:
            raise ValueError("No graph edges supplied; cannot produce measured embeddings")
        ids = torch.tensor(encoded, dtype=torch.long, device=self.device)
        if self.model.relation_types > len(self.relation_to_id):
            edge_index, edge_type = directed_message_graph(ids, len(self.relation_to_id))
        else:
            edge_index, edge_type = message_graph(ids)
        with torch.no_grad():
            return self.model.encode(edge_index, edge_type)

    def refresh(self, graph_triples: list[tuple[str, str, str]]):
        """Re-encode node embeddings for a graph. Call when the adjacency changes."""
        self._embeddings = None
        self._graph_key = None
        key = (len(graph_triples), hash(tuple(graph_triples)))
        self._embeddings = self._encode_graph(graph_triples)
        self._graph_key = key
        return self

    def score_batch(self, triples: list[tuple[str, str, str]]) -> list[float]:
        if self._embeddings is None:
            raise RuntimeError("No graph supplied; call refresh() before scoring.")
        if not triples:
            return []
        rows = []
        for source, relation, target in triples:
            source_index = self.entity_to_id.get(source)
            relation_index = self.relation_to_id.get(relation)
            target_index = self.entity_to_id.get(target)
            if source_index is None or relation_index is None or target_index is None:
                raise ValueError("Unmeasured out-of-vocabulary triple; no neutral measured prior")
            else:
                rows.append([source_index, relation_index, target_index])
        ids = torch.tensor(rows, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model.score(self._embeddings, ids)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite GNN logits")
            scores = torch.sigmoid(logits).tolist()
        return [float(score) for score in scores]

    def score_triple(self, source: str, relation: str, target: str) -> float:
        return self.score_batch([(source, relation, target)])[0]

    @classmethod
    def load(cls, path: str | Path, graph_triples: list[tuple[str, str, str]],
             device: str | None = None) -> "GNNEdgeScorer":
        path = Path(path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config = checkpoint.get("config") or {}
        entity_to_id = checkpoint["entity_to_id"]
        relation_to_id = checkpoint["relation_to_id"]
        model = GNNLinkPredictor(
            len(entity_to_id),
            len(relation_to_id),
            config.get("model", "rgcn"),
            config.get("embedding_dim", config.get("embedding-dim", 64)),
            config.get("layers", 2),
            config.get("dropout", 0.1),
            config.get("decoder", "distmult"),
            bool(config.get("reverse_relations", config.get("reverse-relations", False))),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, entity_to_id, relation_to_id, graph_triples,
                   device=device, checkpoint_name=path.stem)


class FrozenRGCNScorer:
    """Hash-bound export of the trained R-GCN; excludes untrained vocabulary."""

    @classmethod
    def load(cls, manifest_path, expected_sha256):
        self = cls()
        self.path = Path(manifest_path).resolve()
        self.expected = expected_sha256
        self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
        if self.manifest.get('schema') != 'frozen-rgcn-export.v1':
            raise ValueError('Unsupported R-GCN export schema')
        self.root = self.path.parent
        required = {'selected.pt','entity_embeddings.npy','exposure.npz','RESULT.json',
                    'data/vocabulary.json','data/DATASET.json'}
        if set(self.manifest.get('files', {})) != required:
            raise ValueError('Incomplete R-GCN export binding')
        self.verify_files()
        checkpoint = torch.load(self.root/'selected.pt', map_location='cpu', weights_only=True)
        dataset = json.loads((self.root/'data/DATASET.json').read_text(encoding='utf-8'))
        if (checkpoint['graph_revision'] != self.manifest['graph_revision'] or
                dataset['graph_revision'] != self.manifest['graph_revision'] or
                checkpoint['epoch'] != self.manifest['selected_epoch'] or
                checkpoint['execution_sha256'] != self.manifest['execution_sha256'] or
                checkpoint['dataset_sha256'] != self.manifest['dataset_sha256'] or
                self.manifest['dataset_sha256'] != self.manifest['files']['data/DATASET.json']):
            raise ValueError('R-GCN training/export identities disagree')
        config = checkpoint['config']
        if config.get('model') != 'rgcn' or config.get('decoder') != 'directional' or not config.get('reverse_relations'):
            raise ValueError('Expected relational encoder with directed messages and decoder')
        vocabulary = json.loads((self.root/'data/vocabulary.json').read_text(encoding='utf-8'))
        entities,relations = vocabulary['entities'],vocabulary['relations']
        if any(not isinstance(v,str) or not v for v in entities+relations) or len(set(entities))!=len(entities) or len(set(relations))!=len(relations):
            raise ValueError('Invalid export vocabulary')
        self.embeddings = np.load(self.root/'entity_embeddings.npy',mmap_mode='r',allow_pickle=False)
        dimension = config['embedding_dim']
        state = checkpoint['state_dict']
        if config.get('layers') != 2:
            raise ValueError('Expected the two-layer R-GCN export')
        for layer in range(2):
            if (tuple(state[f'layers.{layer}.weight'].shape) != (2*len(relations),dimension,dimension) or
                    tuple(state[f'layers.{layer}.root'].shape) != (dimension,dimension) or
                    tuple(state[f'layers.{layer}.bias'].shape) != (dimension,)):
                raise ValueError('Missing or incompatible relational message-passing weights')
        if (self.embeddings.shape != (len(entities),dimension) or
                tuple(state['entity_embedding.weight'].shape) != self.embeddings.shape):
            raise ValueError('R-GCN embedding dimensions do not match')
        self.matrices = state['decoder.bilinear'].numpy().astype(np.float64)
        self.bias = state['decoder.bias'].numpy().astype(np.float64)
        if self.matrices.shape != (len(relations),dimension,dimension) or self.bias.shape != (len(relations),):
            raise ValueError('R-GCN decoder dimensions do not match')
        with np.load(self.root/'exposure.npz',allow_pickle=False) as masks:
            nodes,edges = masks['trained_nodes'],masks['trained_relations']
        if nodes.dtype != bool or edges.dtype != bool or nodes.shape != (len(entities),) or edges.shape != (len(relations),):
            raise ValueError('Invalid training exposure masks')
        self.ent2idx = {value:i for i,value in enumerate(entities) if nodes[i]}
        self.rel2idx = {value:i for i,value in enumerate(relations) if edges[i]}
        if not np.isfinite(self.embeddings).all() or not np.isfinite(self.matrices).all() or not np.isfinite(self.bias).all():
            raise ValueError('Nonfinite R-GCN export')
        self.name = f"frozen_rgcn_directional:epoch_{checkpoint['epoch']}"
        self.provenance = dict(architecture='rgcn',decoder='directional',direction_sensitive=True,
                              training_graph_revision=checkpoint['graph_revision'],
                              dataset_sha256=checkpoint['dataset_sha256'],selected_epoch=checkpoint['epoch'],
                              inference_message_graph='training_edges_only',embeddings_recomputed=False,
                              artifact_bindings_verified=True,scientific_usefulness_validated=False,
                              untrained_entities_excluded=int((~nodes).sum()))
        self.verify_files()
        return self

    def verify_files(self):
        def digest(path):
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8<<20),b''):
                    h.update(block)
            return h.hexdigest()
        if digest(self.path)!=self.expected:
            raise ValueError('R-GCN export manifest changed')
        for name,expected in self.manifest['files'].items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or digest(path)!=expected:
                raise ValueError(f'R-GCN export file changed: {name}')

    def score_batch(self, triples):
        result=[]
        for source,relation,target in triples:
            if source not in self.ent2idx or target not in self.ent2idx or relation not in self.rel2idx:
                raise ValueError('Unmeasured out-of-vocabulary or untrained triple')
            s=self.embeddings[self.ent2idx[source]].astype(np.float64)
            t=self.embeddings[self.ent2idx[target]].astype(np.float64)
            r=self.rel2idx[relation]
            logit=float((s@self.matrices[r])@t+self.bias[r])
            if not math.isfinite(logit):
                raise ValueError('Nonfinite R-GCN score')
            result.append(1/(1+math.exp(-max(-700.,min(700.,logit)))))
        return result

    def score_triple(self,source,relation,target):
        return self.score_batch([(source,relation,target)])[0]


class FrozenGraphSAGEScorer:
    """Decode the bound, already encoded GraphSAGE export without recomputing its graph.

    These are outputs of the historical GNN, not freshly learned KG embeddings.
    New/OOV nodes remain unmeasured. Training and target KG versions stay distinct.
    """

    name = 'frozen_graphsage:two_layer:distmult'

    @staticmethod
    def _hash(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def verify_files(self):
        for path, expected in self.bindings.items():
            if self._hash(path) != expected:
                raise ValueError(f'Frozen GraphSAGE export fingerprint changed: {path.name}')

    @classmethod
    def load(cls, manifest_path, expected_sha256):
        path = Path(manifest_path).resolve()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('GraphSAGE export manifest fingerprint changed')
        manifest = json.loads(raw)
        root = path.parent.parent
        files = {name.replace('\\', '/'): (root / name).resolve() for name in manifest['files']}
        result = cls()
        result.bindings = {path: expected_sha256, root / 'model/TRAINED.json': manifest['trained_receipt_sha256']}
        result.bindings.update({files[name.replace('\\', '/')]: record['sha256']
                                for name, record in manifest['files'].items()})
        result.verify_files()
        trained = json.loads((root / 'model/TRAINED.json').read_text(encoding='utf-8'))
        if trained.get('actual_model') != 'two_layer_GraphSAGE_with_DistMult':
            raise ValueError('Export does not identify the trained two-layer GraphSAGE')
        checkpoint_key = f"model/weights_epoch_{trained['selected_epoch']:03d}.pt"
        checkpoint = torch.load(files[checkpoint_key], map_location='cpu', weights_only=True)
        for key, recorded in (('kg_sha256', manifest['kg_sha256']), ('lock_id', manifest['training_lock_id'])):
            if checkpoint.get(key) != recorded:
                raise ValueError(f'Checkpoint/export {key} differs')
        if (trained['kg_sha256'] != manifest['kg_sha256'] or
                trained['training_lock_id'] != manifest['training_lock_id'] or
                checkpoint.get('epoch') != trained['selected_epoch']):
            raise ValueError('Training receipt differs from its export')
        vocab = json.loads(files['gnn_inputs/vocabulary.json'].read_text(encoding='utf-8'))
        for name in ('entities', 'relations'):
            values = vocab[name]
            if (not isinstance(values, list) or not values or
                    any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values)):
                raise ValueError('Invalid or duplicated GraphSAGE vocabulary')
        result.ent2idx = {value: i for i, value in enumerate(vocab['entities'])}
        result.rel2idx = {value: i for i, value in enumerate(vocab['relations'])}
        result.entities = np.load(files['model/entity_embeddings.npy'], mmap_mode='r', allow_pickle=False)
        result.relations = np.load(files['model/relation_embeddings.npy'], mmap_mode='r', allow_pickle=False)
        state = checkpoint['state_dict']
        dimension = result.entities.shape[1] if result.entities.ndim == 2 else 0
        if any(key not in state or tuple(state[key].shape) != (dimension, dimension)
               for key in (f'layers.{i}.{side}.weight' for i in range(2) for side in ('lin_l', 'lin_r'))):
            raise ValueError('Checkpoint does not contain the expected two-layer GraphSAGE weights')
        if (result.entities.ndim != 2 or result.relations.ndim != 2 or
                result.entities.shape[0] != len(result.ent2idx) or result.relations.shape[0] != len(result.rel2idx) or
                result.entities.shape[1] != result.relations.shape[1] or
                tuple(state['entity_embedding.weight'].shape) != result.entities.shape or
                not np.array_equal(state['relation_embedding.weight'].numpy(), result.relations)):
            raise ValueError('GNN vocabulary, checkpoint and export dimensions/relations differ')
        result.provenance = dict(architecture='two_layer_GraphSAGE', decoder='DistMult', direction_sensitive=False,
            training_graph_revision=manifest['kg_sha256'], training_lock_id=manifest['training_lock_id'],
            checkpoint_sha256=result.bindings[files[checkpoint_key]], export_manifest_sha256=expected_sha256,
            selected_epoch=trained['selected_epoch'], embeddings_recomputed=False,
            artifact_bindings_verified=True, frozen_execution_audit_revalidated=False,
            scientific_usefulness_validated=False)
        result.verify_files()
        return result

    def score_batch(self, triples):
        values = []
        for source, relation, target in triples:
            if source not in self.ent2idx or target not in self.ent2idx or relation not in self.rel2idx:
                raise ValueError('Unmeasured out-of-vocabulary triple; no neutral measured prior')
            # Match the frozen decoder's float64 accumulation and sigmoid exactly.
            logit = float(np.sum(self.entities[self.ent2idx[source]].astype(np.float64) *
                                 self.relations[self.rel2idx[relation]] * self.entities[self.ent2idx[target]]))
            if not math.isfinite(logit):
                raise ValueError('Nonfinite GNN export score')
            values.append(1 / (1 + math.exp(-max(-700, min(700, logit)))))
        return values
