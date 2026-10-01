"""Offline tests for the GNN -> LinkScorer adapter.

Small synthetic graph, deterministic weights, no training and no real KG.
"""

from __future__ import annotations

import torch
import pytest

from models.kg_link_prediction.gnn import GNNLinkPredictor
from models.kg_link_prediction.scorer import GNNEdgeScorer
from neurooracle.src.kge import hypothesis_path as hp


GRAPH = [("A", "activates", "B"), ("B", "activates", "C")]


def _scorer(**kwargs) -> GNNEdgeScorer:
    torch.manual_seed(0)
    entity_to_id = {"A": 0, "B": 1, "C": 2}
    relation_to_id = {"activates": 0}
    model = GNNLinkPredictor(3, 1, "rgcn", embedding_dim=8, layers=1, **kwargs)
    return GNNEdgeScorer(model, entity_to_id, relation_to_id, GRAPH)


def test_adapter_exposes_the_mapping_surface_hypothesis_path_expects():
    scorer = _scorer()
    assert scorer.ent2idx == {"A": 0, "B": 1, "C": 2}
    assert scorer.rel2idx == {"activates": 0}
    assert scorer.score_batch([("A", "activates", "B")])[0] == scorer.score_triple("A", "activates", "B")


def test_out_of_vocabulary_triples_are_unmeasured():
    scorer = _scorer()
    with pytest.raises(ValueError, match="Unmeasured"):
        scorer.score_batch([("A", "activates", "ZZZ")])


@pytest.mark.parametrize("graph", [[], [("A", "activates", "ZZZ")]])
def test_invalid_graph_refresh_cannot_leave_old_embeddings_active(graph):
    scorer = _scorer()
    with pytest.raises(ValueError):
        scorer.refresh(graph)
    with pytest.raises(RuntimeError):
        scorer.score_batch(GRAPH)


def test_scores_are_finite_probabilities():
    scorer = _scorer()
    scores = scorer.score_batch([("A", "activates", "B"), ("B", "activates", "C")])
    assert all(0.0 <= value <= 1.0 for value in scores)


def test_empty_batch_is_empty_not_an_error():
    assert _scorer().score_batch([]) == []


def test_scoring_without_a_graph_fails_closed():
    scorer = _scorer()
    scorer._embeddings = None
    try:
        scorer.score_batch([("A", "activates", "B")])
    except RuntimeError as exc:
        assert "refresh" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a fail-closed RuntimeError")


def test_adapter_plugs_into_hypothesis_path_without_policy_changes():
    scorer = _scorer()
    result = hp.score_candidate(
        {"kg_triples": [{"source": "A", "relation": "activates", "target": "B"}]},
        scorer=scorer,
        edge_confidence=lambda *triple: 0.9,
        pair_support=lambda *pair: 1,
        provenance_fraction=1.0,
        testability=0.8,
    )
    assert result["measured"] is True
    assert 0.0 <= result["gnn_path_score"] <= 1.0
    assert result["asserts_truth_or_novelty"] is False


@pytest.mark.parametrize('architecture', ['rgcn', 'graphsage'])
def test_idea_manifest_loads_real_gnn_inference_and_rechecks_cached_bindings(tmp_path, monkeypatch, architecture):
    """Synthetic untrained weights prove wiring only, never hypothesis quality."""
    import hashlib
    import json
    from types import SimpleNamespace
    from core.idea_ranking import configured_scorer
    model = GNNLinkPredictor(3, 1, architecture, embedding_dim=8, layers=1, decoder='bilinear', reverse_relations=True)
    checkpoint, graph, manifest = [tmp_path / name for name in ('weights.pt', 'graph.json', 'manifest.json')]
    torch.save(dict(state_dict=model.state_dict(), entity_to_id={'A': 0, 'B': 1, 'C': 2},
                    relation_to_id={'activates': 0}, config=dict(model=architecture, embedding_dim=8, layers=1,
                    decoder='bilinear', reverse_relations=True)), checkpoint)
    graph.write_text(json.dumps(GRAPH), encoding='utf-8')
    def binding(path):
        return dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest.write_text(json.dumps(dict(graph_revision='synthetic', training_scope='Synthetic, untrained fixture only',
                        checkpoint=binding(checkpoint), message_graph=binding(graph))), encoding='utf-8')
    monkeypatch.setenv('NEUROCLAW_IDEA_GNN_MANIFEST', str(manifest))
    layer = SimpleNamespace()
    scorer, receipt = configured_scorer(layer, {'graph_revision': 'synthetic'})
    assert isinstance(scorer, GNNEdgeScorer) and receipt['status'] == 'available'
    assert len(scorer.score_batch(GRAPH)) == 2
    assert configured_scorer(layer, {'graph_revision': 'synthetic'})[0] is scorer
    graph.write_text('[]', encoding='utf-8')
    missing, failure = configured_scorer(layer, {'graph_revision': 'synthetic'})
    assert missing is None and 'fingerprint' in failure['reason']


def _frozen_export(tmp_path):
    """Tiny untrained GraphSAGE export, not a scientific model or training run."""
    import json
    import hashlib
    import numpy as np
    model_dir = tmp_path / 'model'
    inputs = tmp_path / 'gnn_inputs'
    model_dir.mkdir(); inputs.mkdir()
    model = GNNLinkPredictor(3, 1, 'graphsage', embedding_dim=4, layers=2)
    scorer = GNNEdgeScorer(model, {'A': 0, 'B': 1, 'C': 2}, {'activates': 0}, GRAPH)
    torch.save(dict(state_dict=model.state_dict(), epoch=1, kg_sha256='old-graph', lock_id='original-lock'),
               model_dir / 'weights_epoch_001.pt')
    np.save(model_dir / 'entity_embeddings.npy', scorer._embeddings.numpy())
    np.save(model_dir / 'relation_embeddings.npy', model.relation_embedding.weight.detach().numpy())
    (inputs / 'vocabulary.json').write_text(json.dumps(dict(entities=['A', 'B', 'C'], relations=['activates'])), encoding='utf-8')
    trained = model_dir / 'TRAINED.json'
    trained.write_text(json.dumps(dict(actual_model='two_layer_GraphSAGE_with_DistMult', selected_epoch=1,
                       kg_sha256='old-graph', training_lock_id='original-lock')), encoding='utf-8')
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = model_dir / 'inference_manifest.json'
    manifest.write_text(json.dumps(dict(kg_sha256='old-graph', training_lock_id='original-lock',
        trained_receipt_sha256=sha(trained), files={str(p.relative_to(tmp_path)): {'sha256': sha(p)} for p in (
            model_dir / 'weights_epoch_001.pt', model_dir / 'entity_embeddings.npy',
            model_dir / 'relation_embeddings.npy', inputs / 'vocabulary.json')})), encoding='utf-8')
    return manifest, sha, scorer


def test_frozen_graphsage_scores_match_encoder_and_decoder_and_never_fill_oov(tmp_path):
    from models.kg_link_prediction.scorer import FrozenGraphSAGEScorer
    manifest, sha, original = _frozen_export(tmp_path)
    frozen = FrozenGraphSAGEScorer.load(manifest, sha(manifest))
    assert frozen.score_batch(GRAPH) == pytest.approx(original.score_batch(GRAPH), abs=1e-7)
    assert frozen.provenance['training_graph_revision'] == 'old-graph'
    assert not frozen.provenance['direction_sensitive']
    assert not frozen.provenance['frozen_execution_audit_revalidated']
    assert not frozen.entities.flags.writeable and not frozen.relations.flags.writeable
    assert frozen.score_batch([('B', 'activates', 'A')]) == pytest.approx(frozen.score_batch([GRAPH[0]]))
    with pytest.raises(ValueError, match='out-of-vocabulary'):
        frozen.score_batch([('A', 'activates', 'new-node')])


@pytest.mark.parametrize('relative', ['model/weights_epoch_001.pt', 'model/entity_embeddings.npy',
                                     'model/relation_embeddings.npy', 'gnn_inputs/vocabulary.json', 'model/TRAINED.json'])
def test_frozen_graphsage_cache_rejects_tampered_inputs(tmp_path, relative):
    from models.kg_link_prediction.scorer import FrozenGraphSAGEScorer
    manifest, sha, original = _frozen_export(tmp_path)
    frozen = FrozenGraphSAGEScorer.load(manifest, sha(manifest))
    with (tmp_path / relative).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='fingerprint'):
        frozen.verify_files()


def test_idea_graphsage_export_keeps_old_training_revision_separate(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from core.idea_ranking import configured_scorer
    manifest, sha, original = _frozen_export(tmp_path)
    config = tmp_path / 'idea_gnn.json'
    config.write_text(json.dumps(dict(kind='graphsage_export', graph_revision='current-graph',
        training_scope='Synthetic legacy transfer, no training',
        export=dict(path=str(manifest), sha256=sha(manifest)))), encoding='utf-8')
    monkeypatch.setenv('NEUROCLAW_IDEA_GNN_MANIFEST', str(config))
    layer = SimpleNamespace()
    scorer, receipt = configured_scorer(layer, {'graph_revision': 'current-graph'})
    assert scorer is not None and receipt['artifact_bindings_verified']
    assert receipt['training_graph_revision'] == 'old-graph' and not receipt['trained_on_current_graph']
    assert configured_scorer(layer, {'graph_revision': 'current-graph'})[0] is scorer
    wrong, error = configured_scorer(layer, {'graph_revision': 'another-graph'})
    assert wrong is None and 'different graph' in error['reason']
