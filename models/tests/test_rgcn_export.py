import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.kg_link_prediction.full_graph_data import sha,write
from models.kg_link_prediction.gnn import GNNLinkPredictor,directed_message_graph
from models.kg_link_prediction.scorer import FrozenRGCNScorer


def export_fixture(root):
    (root/'data').mkdir()
    config=dict(model='rgcn',embedding_dim=8,layers=2,dropout=0.,decoder='directional',reverse_relations=True)
    torch.manual_seed(7)
    model=GNNLinkPredictor(5,2,**config).eval()
    triples=torch.tensor([[0,0,1],[1,1,2],[2,0,3]])
    graph,kinds=directed_message_graph(triples,2)
    with torch.no_grad():
        z=model.encode(graph,kinds)
    np.save(root/'entity_embeddings.npy',z.numpy(),allow_pickle=False)
    np.savez(root/'exposure.npz',trained_nodes=np.array([True,True,True,True,False]),trained_relations=np.array([True,True]))
    write(root/'data/vocabulary.json',dict(entities=['a','b','c','d','untrained'],relations=['r0','r1']))
    write(root/'data/DATASET.json',dict(graph_revision='graph'))
    dataset_sha=sha(root/'data/DATASET.json')
    torch.save(dict(state_dict=model.state_dict(),config=config,epoch=3,graph_revision='graph',
                    dataset_sha256=dataset_sha,execution_sha256='execution'),root/'selected.pt')
    write(root/'RESULT.json',dict(selected_epoch=3))
    write(root/'inference_manifest.json',dict(schema='frozen-rgcn-export.v1',graph_revision='graph',
          selected_epoch=3,dataset_sha256=dataset_sha,execution_sha256='execution',
          files={name:sha(root/name) for name in ['selected.pt','entity_embeddings.npy','exposure.npz',
                   'RESULT.json','data/vocabulary.json','data/DATASET.json']}))
    return model,z


def test_export_decoder_agrees_with_checkpoint_and_excludes_untrained(tmp_path):
    model,z=export_fixture(tmp_path)
    scorer=FrozenRGCNScorer.load(tmp_path/'inference_manifest.json',sha(tmp_path/'inference_manifest.json'))
    expected=torch.sigmoid(model.score(z,torch.tensor([[0,0,1],[1,0,0],[2,1,3]]))).detach().numpy()
    actual=scorer.score_batch([('a','r0','b'),('b','r0','a'),('c','r1','d')])
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
    assert 'untrained' not in scorer.ent2idx
    with pytest.raises(ValueError,match='untrained'):
        scorer.score_batch([('a','r0','untrained')])
    assert scorer.provenance['direction_sensitive']


def test_export_detects_file_drift_and_false_manifest(tmp_path):
    export_fixture(tmp_path)
    path=tmp_path/'inference_manifest.json'
    with pytest.raises(ValueError,match='manifest changed'):
        FrozenRGCNScorer.load(path,'0'*64)
    scorer=FrozenRGCNScorer.load(path,sha(path))
    with (tmp_path/'exposure.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='file changed'):
        scorer.verify_files()


def test_idea_runtime_loads_bound_rgcn_export_and_cache_checks(tmp_path,monkeypatch):
    from core.idea_ranking import configured_scorer
    export_fixture(tmp_path)
    manifest=tmp_path/'idea.json'
    write(manifest,dict(kind='rgcn_export',graph_revision='graph',training_scope='Synthetic interface test',
                       export=dict(path='inference_manifest.json',sha256=sha(tmp_path/'inference_manifest.json'))))
    monkeypatch.setenv('NEUROCLAW_IDEA_GNN_MANIFEST',str(manifest))
    layer=SimpleNamespace()
    scorer,receipt=configured_scorer(layer,dict(graph_revision='graph'))
    assert scorer is not None and receipt['trained_on_current_graph']
    assert receipt['direction_sensitive']
    with (tmp_path/'exposure.npz').open('ab') as stream:
        stream.write(b'changed')
    assert configured_scorer(layer,dict(graph_revision='graph'))[0] is None


def test_independent_numpy_encoder_matches_two_layer_pyg(tmp_path):
    from models.kg_link_prediction.full_graph_verify import numpy_encode_subset
    model,z=export_fixture(tmp_path)
    triples=np.array([[0,0,1],[1,1,2],[2,0,3]])
    ids,actual,context=numpy_encode_subset(model.state_dict(),triples,[0,2,4],2)
    np.testing.assert_allclose(actual,z.detach().numpy()[ids],atol=1e-6,rtol=1e-5)
    assert context==5
