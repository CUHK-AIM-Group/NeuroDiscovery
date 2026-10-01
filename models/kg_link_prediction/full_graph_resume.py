"""Restore a committed epoch without resetting its data, patience or RNG state.

Recovery writes to a new, explicitly bound run directory. A completed trajectory
can be inspected/restored but cannot be restarted as another experiment.
"""
from __future__ import annotations

import pyarrow  # noqa: F401
import argparse
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import psutil
import torch

from .full_graph_data import ROOT, read, sha, write
from .gnn import GNNLinkPredictor


def stop_reason(payload):
    config = payload['config']
    if payload['epoch'] >= config['epochs']:
        return 'maximum_epochs'
    if payload['epoch'] >= config['minimum_epochs'] and payload['stale'] >= config['patience']:
        return 'validation_early_stop'
    return None


def load_committed(run, authorization='AUTHORIZATION.json'):
    run = Path(run).resolve()
    auth = read(run/authorization)
    manifest = read(run/'data/DATASET.json')
    if sha(run/'data/DATASET.json') != auth['dataset_sha256']:
        raise ValueError('Resume dataset binding changed')
    checkpoint_sha = sha(run/'resume.pt')
    state = torch.load(run/'resume.pt', map_location='cpu', weights_only=True)
    execution = auth.get('trajectory_execution_sha256', auth['execution_sha256'])
    expected = dict(config=auth['config'], graph_revision=manifest['graph_revision'],
                    dataset_sha256=auth['dataset_sha256'], execution_sha256=execution)
    if any(state.get(k) != v for k, v in expected.items()):
        raise ValueError('Resume checkpoint identity/config mismatch')
    epoch = state['epoch']
    if not isinstance(epoch, int) or not 1 <= epoch <= state['config']['epochs']:
        raise ValueError('Invalid committed epoch')
    history = []
    for i in range(1, epoch+1):
        path = run/'epochs'/f'{i:03}.json'
        record = read(path) if path.exists() else state.get('epoch_record') if i == epoch else None
        if not record or record['epoch'] != i:
            raise ValueError('Missing committed epoch record')
        if i == epoch and 'epoch_record' in state and record != state['epoch_record']:
            raise ValueError('Checkpoint/journal mismatch')
        history.append(record)
    best, best_epoch, stale = -1., 0, 0
    for row in history:
        value = row['validation']['sampled_mrr']
        if value > best+1e-7:
            best, best_epoch, stale = value, row['epoch'], 0
        else:
            stale += 1
    if (best, best_epoch, stale) != (state['best'], state['best_epoch'], state['stale']):
        raise ValueError('Resume patience/validation history mismatch')
    steps = [int(v['step']) for v in state['optimizer']['state'].values()]
    if not steps or set(steps) != {epoch}:
        raise ValueError('Resume optimizer counters disagree with epoch')
    selected = state.get('selected_checkpoint')
    if selected is None:  # Original v1 checkpoints were committed before this repair.
        selected = torch.load(run/'selected.pt', map_location='cpu', weights_only=True)
    if selected['epoch'] != best_epoch or any(selected.get(k) != v for k, v in expected.items()):
        raise ValueError('Selected checkpoint is not the committed validation winner')
    if sha(run/'resume.pt') != checkpoint_sha:
        raise ValueError('Resume checkpoint changed during load')
    return state, selected, history, checkpoint_sha


def restore_training_state(payload, model, optimizer, rng, *, restore_cuda=True):
    model.load_state_dict(payload['state_dict'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    rng.bit_generator.state = payload['rng_numpy']
    torch.set_rng_state(payload['rng_torch'])
    if restore_cuda:
        if len(payload['rng_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed; RNG replay cannot be guaranteed')
        torch.cuda.set_rng_state_all(payload['rng_cuda'])
    if 'rng_python' in payload:
        random.setstate(payload['rng_python'])
    # The original loop never consumes Python/global NumPy random numbers;
    # sampling uses only the restored Generator and Torch CPU/CUDA generators.
    return payload['epoch']+1, payload['best'], payload['best_epoch'], payload['stale']


def prepare_resume(run, auth):
    """Preflight before reserving a new segment; preserve the old attempt verbatim."""
    run = Path(run).resolve()
    binding = auth['resume_from']
    parent = Path(binding['run']).resolve()
    if parent == run or not parent.is_relative_to(ROOT/'tmp'):
        raise ValueError('Resume source must be a separate workspace run')
    # Never steal an in-flight checkpoint. PID reuse fails conservatively.
    for p in parent.glob('PROCESS*.json'):
        pid = read(p).get('worker_pid')
        if pid and psutil.pid_exists(pid):
            raise ValueError('Resume source worker may still be running')
    for name, digest in binding['files'].items():
        source = (parent/name).resolve()
        if not source.is_relative_to(parent) or sha(source) != digest:
            raise ValueError(f'Resume source binding changed: {name}')
    if not {'resume.pt', 'AUTHORIZATION.json', 'data/DATASET.json', 'EVALUATION_CANDIDATES.npz'} <= set(binding['files']):
        raise ValueError('Incomplete resume source binding')
    payload, selected, history, _ = load_committed(parent)
    if (parent/'RESULT.json').exists():
        raise ValueError('Completed trajectory must not be resumed')
    required = {'selected.pt'} | {f'epochs/{row["epoch"]:03}.json' for row in history
                                  if (parent/'epochs'/f'{row["epoch"]:03}.json').exists()}
    if not required <= set(binding['files']):
        raise ValueError('Resume selection and committed logs must be hash-bound')
    parent_auth = read(parent/'AUTHORIZATION.json')
    if auth['config'] != payload['config'] or auth['dataset_sha256'] != payload['dataset_sha256']:
        raise ValueError('Resume cannot change the model, hyperparameters or split')
    if auth.get('trajectory_execution_sha256') != payload['execution_sha256']:
        raise ValueError('Resume must retain the original trajectory identity')
    deadline = parent_auth['original_deadline_unix']
    if (auth.get('original_deadline_unix') != deadline or auth['max_run_seconds'] > deadline-time.time()
            or auth['prior_runs'] < parent_auth['prior_runs']+1):
        raise ValueError('Resume cannot reset the deadline or prior run accounting')
    manifest = read(parent/'data/DATASET.json')
    for name, digest in manifest['files'].items():
        if sha(parent/'data'/name) != digest:
            raise ValueError('Resume data content changed')
    (run/'data').mkdir()
    for name in [*manifest['files'], 'DATASET.json']:
        shutil.copyfile(parent/'data'/name, run/'data'/name)
    shutil.copyfile(parent/'EVALUATION_CANDIDATES.npz', run/'EVALUATION_CANDIDATES.npz')
    (run/'epochs').mkdir()
    for row in history:
        write(run/'epochs'/f"{row['epoch']:03}.json", row)
    torch.save(selected, run/'selected.pt')
    torch.save(payload, run/'resume.pt')
    write(run/'RESUME_SOURCE.json', dict(parent=str(parent), binding=binding,
          committed_epoch=payload['epoch'], next_epoch=payload['epoch']+1,
          uncommitted_parent_logs=[p.name for p in (parent/'epochs').glob('*.json') if int(p.stem)>payload['epoch']],
          parent_modified=False))


def inspect(run, output):
    """Actually load the real model + AdamW state; no optimizer update or GPU allocation."""
    started = time.monotonic()
    torch.set_num_threads(4)
    run = Path(run).resolve()
    payload, selected, history, digest = load_committed(run)
    config = payload['config']
    shape = payload['state_dict']['entity_embedding.weight'].shape
    relations = len(payload['state_dict']['decoder.bias'])
    model = GNNLinkPredictor(shape[0], relations, **{k: config[k] for k in
        ('model', 'embedding_dim', 'layers', 'dropout', 'decoder', 'reverse_relations')})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng = np.random.default_rng()
    next_epoch, best, best_epoch, stale = restore_training_state(payload, model, optimizer, rng, restore_cuda=False)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, payload['state_dict'][name], rtol=0, atol=0)
    loaded_optimizer = optimizer.state_dict()
    for key, state in payload['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(loaded_optimizer['state'][key][name], value, rtol=0, atol=0)
    expected_rng = np.random.default_rng()
    expected_rng.bit_generator.state = payload['rng_numpy']
    np.testing.assert_array_equal(rng.integers(0, 2**31, 32), expected_rng.integers(0, 2**31, 32))
    torch.testing.assert_close(torch.get_rng_state(), payload['rng_torch'], rtol=0, atol=0)
    result = dict(status='REAL_CHECKPOINT_RESTORE_PASSED', resume_sha256=digest, epoch=payload['epoch'],
                  optimizer_steps=sorted({int(v['step']) for v in optimizer.state.values()}),
                  selected_epoch=best_epoch, stale=stale, validation_best=best, restored_model_and_optimizer=True,
                  cpu_and_numpy_rng_restored=True, cuda_rng_saved_devices=len(payload['rng_cuda']),
                  cuda_rng_restored_in_this_cpu_inspection=False, stop_reason=stop_reason(payload),
                  next_training_epoch=None if stop_reason(payload) else next_epoch,
                  new_optimizer_steps=0, input_modified=False, seconds=time.monotonic()-started,
                  code_sha256=sha(__file__))
    write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    inspect(args.run, args.output)
