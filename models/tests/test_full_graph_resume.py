import copy
import os
import random
import time

import numpy as np
import pytest
import torch

from models.kg_link_prediction import full_graph_resume as recovery
from models.kg_link_prediction.full_graph_data import read, sha, write
from models.kg_link_prediction.full_graph_train import optimize_epoch, save_checkpoint
from models.kg_link_prediction.gnn import GNNLinkPredictor
from models.kg_link_prediction.sparse_rgcn import relation_blocks, triple_keys


def fixture_run(run, device='cpu'):
    (run/'data').mkdir(parents=True)
    (run/'epochs').mkdir()
    config = dict(model='rgcn', embedding_dim=8, layers=2, dropout=.2, decoder='directional',
                  reverse_relations=True, seed=19, epochs=6, minimum_epochs=2, patience=2,
                  learning_rate=.001, weight_decay=.00001, decoder_batch_size=2)
    write(run/'data/DATASET.json', dict(graph_revision='synthetic', files={}))
    np.savez(run/'EVALUATION_CANDIDATES.npz', rows=np.array([[0, 0, 1]]))
    auth = dict(config=config, graph_revision='synthetic', dataset_sha256=sha(run/'data/DATASET.json'),
                execution_sha256='original-trajectory', original_deadline_unix=time.time()+600,
                prior_runs=0, max_runs=2, max_run_seconds=300)
    write(run/'AUTHORIZATION.json', auth)
    torch.manual_seed(19)
    if device == 'cuda':
        torch.cuda.manual_seed_all(19)
    model = GNNLinkPredictor(12, 2, **{k: config[k] for k in
                ('model', 'embedding_dim', 'layers', 'dropout', 'decoder', 'reverse_relations')}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
    rng = np.random.default_rng(19)
    triples = np.array([[0, 0, 1], [2, 1, 3], [4, 0, 5], [6, 1, 7], [8, 0, 9], [10, 1, 11]])
    blocks = relation_blocks(triples, 12, 2, device)
    domains, trained = np.zeros(12, dtype=np.int16), np.ones(12, dtype=bool)
    known = np.sort(triple_keys(triples, 12, 2))
    def step(m=model, opt=optimizer, gen=rng):
        return optimize_epoch(m, opt, blocks, triples, domains, trained, known, gen, device, config)
    loss, _, _ = step()
    record = dict(epoch=1, validation=dict(sampled_mrr=.2), positive_count=len(triples), loss=loss)
    selected = dict(state_dict=copy.deepcopy(model.cpu().state_dict()), config=config, epoch=1,
                    graph_revision='synthetic', dataset_sha256=auth['dataset_sha256'],
                    execution_sha256=auth['execution_sha256'])
    model.to(device)
    payload = copy.deepcopy(selected)
    payload.update(optimizer=copy.deepcopy(optimizer.state_dict()), rng_numpy=copy.deepcopy(rng.bit_generator.state),
                   rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else [],
                   rng_python=random.getstate(), best=.2, best_epoch=1, stale=0, epoch_record=record,
                   selected_checkpoint=selected)
    save_checkpoint(run/'resume.pt', payload)
    save_checkpoint(run/'selected.pt', selected)
    write(run/'epochs/001.json', record)
    return model, optimizer, rng, step, auth


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_interrupted_training_matches_uninterrupted_model_optimizer_rng(tmp_path, device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(2)
    run = tmp_path/'run'
    model, optimizer, rng, step, auth = fixture_run(run, device)
    expected_losses = [step()[0], step()[0]]
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_numpy = rng.integers(0, 2**31, 10)
    expected_torch = torch.rand(10, device=device)
    # Deliberately perturb all relevant state as a new process would.
    torch.manual_seed(999)
    if device == 'cuda':
        torch.cuda.manual_seed_all(999)
    replacement = GNNLinkPredictor(12, 2, 'rgcn', 8, 2, .2, 'directional', True).to(device)
    replacement_opt = torch.optim.AdamW(replacement.parameters(), lr=.9)
    replacement_rng = np.random.default_rng(999)
    payload, _, history, _ = recovery.load_committed(run)
    assert recovery.restore_training_state(payload, replacement, replacement_opt, replacement_rng,
                                           restore_cuda=device=='cuda') == (2, .2, 1, 0)
    actual_losses = [step(replacement, replacement_opt, replacement_rng)[0] for _ in range(2)]
    np.testing.assert_allclose(actual_losses, expected_losses, rtol=1e-6, atol=1e-7)
    for key, value in replacement.state_dict().items():
        torch.testing.assert_close(value, expected_model[key], rtol=1e-6, atol=1e-7)
    for key, row in replacement_opt.state_dict()['state'].items():
        for name, value in row.items():
            torch.testing.assert_close(value, expected_optimizer['state'][key][name], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(replacement_rng.integers(0, 2**31, 10), expected_numpy)
    torch.testing.assert_close(torch.rand(10, device=device), expected_torch, rtol=0, atol=0)
    assert len(history)==1


def test_committed_payload_recovers_missing_last_log_and_selected_file(tmp_path):
    run=tmp_path/'run'
    fixture_run(run)
    (run/'epochs/001.json').unlink()
    (run/'selected.pt').write_bytes(b'interrupted next selection')
    payload, selected, history, _ = recovery.load_committed(run)
    assert selected['epoch']==history[0]['epoch']==payload['epoch']==1


@pytest.mark.parametrize('field', ['stale', 'epoch', 'config', 'optimizer'])
def test_corrupt_resume_identity_or_counters_rejected(tmp_path,field):
    run=tmp_path/'run'
    fixture_run(run)
    payload=torch.load(run/'resume.pt',weights_only=True)
    if field=='config':
        payload[field]['dropout']=.9
    elif field=='optimizer':
        next(iter(payload[field]['state'].values()))['step']=torch.tensor(5.)
    else:
        payload[field]+=1
    save_checkpoint(run/'resume.pt',payload)
    with pytest.raises(ValueError):
        recovery.load_committed(run)


def bound_recovery(parent, auth):
    files=['resume.pt','selected.pt','AUTHORIZATION.json','data/DATASET.json',
           'EVALUATION_CANDIDATES.npz','epochs/001.json']
    return dict(auth, prior_runs=1, max_run_seconds=60, trajectory_execution_sha256=auth['execution_sha256'],
                resume_from=dict(run=str(parent),files={p:sha(parent/p) for p in files}))


def test_recovery_preflight_preserves_parent_and_rejects_budget_reset(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    parent=tmp_path/'tmp'/'parent'
    _,_,_,_,auth=fixture_run(parent)
    bound=bound_recovery(parent,auth)
    destination=tmp_path/'tmp'/'recovery'
    destination.mkdir()
    recovery.prepare_resume(destination,bound)
    assert read(destination/'RESUME_SOURCE.json')['next_epoch']==2
    assert sha(parent/'resume.pt')==bound['resume_from']['files']['resume.pt']
    for field,value in [('prior_runs',0),('original_deadline_unix',time.time()+900)]:
        invalid=dict(bound,**{field:value})
        with pytest.raises(ValueError,match='reset'):
            recovery.prepare_resume(tmp_path/'tmp'/field,invalid)


def test_completed_run_is_not_restarted(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    parent=tmp_path/'tmp'/'parent'
    _,_,_,_,auth=fixture_run(parent)
    write(parent/'RESULT.json',dict(status='finished'))
    with pytest.raises(ValueError,match='Completed trajectory'):
        recovery.prepare_resume(tmp_path/'tmp'/'unused',bound_recovery(parent,auth))
    assert not (tmp_path/'tmp'/'unused').exists()


def test_stop_decision_retains_validation_patience():
    payload=dict(epoch=4,stale=2,config=dict(epochs=50,minimum_epochs=2,patience=2))
    assert recovery.stop_reason(payload)=='validation_early_stop'
    payload['epoch']=50
    assert recovery.stop_reason(payload)=='maximum_epochs'
    payload.update(epoch=1,stale=0)
    assert recovery.stop_reason(payload) is None
