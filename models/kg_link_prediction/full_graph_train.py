"""One accounted full-graph R-GCN reconstruction run; no scientific-label claims."""
from __future__ import annotations

# Avoid the local Arrow/native-loader initialization conflict before PyG imports.
import pyarrow  # noqa: F401
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import psutil
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .full_graph_data import ROOT, fingerprint, read, sha, write
from .gnn import GNNLinkPredictor
from .run_guard import RunGuard
from .sparse_rgcn import encode_sparse, relation_blocks, sample_typed_negatives, triple_keys
from .full_graph_resume import load_committed, prepare_resume, restore_training_state, stop_reason

CONFIG = dict(model='rgcn', embedding_dim=64, layers=2, dropout=.1, decoder='directional',
              reverse_relations=True, seed=123, epochs=50, minimum_epochs=10, patience=5,
              learning_rate=.001, weight_decay=.00001, negatives=1, decoder_batch_size=8192,
              ranking_queries=512, ranking_contrasts=100, optimizer_steps_per_epoch=1)
CODE = [Path(__file__), Path(__file__).with_name('full_graph_data.py'),
        Path(__file__).with_name('sparse_rgcn.py'), Path(__file__).with_name('gnn.py'),
        Path(__file__).with_name('run_guard.py'), Path(__file__).with_name('full_graph_resume.py')]


def code_binding():
    return {str(p.relative_to(ROOT)): sha(p) for p in CODE}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return value


def save_checkpoint(path, payload):
    temporary = path.with_name(path.name + '.writing')
    torch.save(payload, temporary)
    temporary.replace(path)


def ranking(scores):
    correct = scores[:, :1]
    tied = np.isclose(scores[:, 1:], correct, atol=1e-6, rtol=1e-5)
    ranks = 1 + ((scores[:, 1:] > correct) & ~tied).sum(axis=1) + .5 * tied.sum(axis=1)
    return dict(sampled_mrr=float(np.mean(1/ranks)), hits_at_1=float(np.mean(ranks == 1)),
                hits_at_10=float(np.mean(ranks <= 10)), queries=len(scores),
                contrasts_per_query=scores.shape[1]-1, all_entity_filtered_mrr=False)


def optimize_epoch(model, optimizer, blocks, train_ids, domains, trained_nodes, known, rng,
                   device, config, check=lambda: None):
    """One full positive pass, shared by initial training and exact-state recovery."""
    optimizer.zero_grad(set_to_none=True)
    model.train()
    contrasts, contrast_mask = sample_typed_negatives(
        train_ids, domains, trained_nodes, known, len(model.decoder.bias), rng, allow_missing=True)
    n_contrasts = int(contrast_mask.sum())
    if not n_contrasts:
        raise ValueError('No reconstruction contrasts available')
    embeddings = encode_sparse(model, blocks, checkpoint_layers=True)
    leaf = embeddings.detach().requires_grad_(True)
    loss_sum = 0.
    batch = config['decoder_batch_size']
    for first in range(0, len(train_ids), batch):
        last = first + batch
        pos = model.score(leaf, torch.as_tensor(train_ids[first:last], device=device))
        neg = model.score(leaf, torch.as_tensor(contrasts[first:last][contrast_mask[first:last]], device=device))
        loss = (torch.nn.functional.softplus(-pos).sum()/len(train_ids)
                +torch.nn.functional.softplus(neg).sum()/n_contrasts)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        loss_sum += float(loss.detach())
        if first % (batch*32) == 0:
            check()
    embeddings.backward(leaf.grad)
    grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss_sum, grad, n_contrasts


def train(run, attempt, authorization='AUTHORIZATION.json'):
    auth = read(run / authorization)
    guard = RunGuard(Path(auth['ledger_path']), auth)
    seconds = guard.claim_worker(attempt, auth['execution_sha256'])
    started = time.monotonic()
    process = psutil.Process()
    peak_cpu = 0
    if code_binding() != auth['code']:
        raise ValueError('Training code changed since authorization binding')
    guard.configure_cuda(torch)
    torch.set_num_threads(4)
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])
    # Numerical reproducibility over speed; do not change math after timing it.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = 'cuda:0'

    def check():
        nonlocal peak_cpu
        peak_cpu = max(peak_cpu, process.memory_info().rss)
        guard.check_usage(elapsed=time.monotonic()-started,
                          gpu_bytes=torch.cuda.max_memory_reserved(), cpu_bytes=peak_cpu,
                          reserved_seconds=seconds)

    def progress(stage, **kw):
        check()
        result = dict(stage=stage, pid=os.getpid(), elapsed_seconds=time.monotonic()-started,
                      gpu_allocated_GiB=torch.cuda.memory_allocated()/2**30,
                      gpu_reserved_GiB=torch.cuda.memory_reserved()/2**30, **kw)
        write(run / 'PROGRESS.json', result, replace=True)
        print(json.dumps(result), flush=True)

    data_dir = run / 'data'
    manifest = read(data_dir / 'DATASET.json')
    if sha(data_dir / 'DATASET.json') != auth['dataset_sha256']:
        raise ValueError('Dataset manifest changed')
    for name, expected in manifest['files'].items():
        if sha(data_dir / name) != expected:
            raise ValueError(f'Dataset changed: {name}')
    data = np.load(data_dir / 'data.npz', allow_pickle=False)
    triples, split = data['triples'], data['split']
    domains, trained_nodes, trained_relations = data['domain_ids'], data['trained_nodes'], data['trained_relations']
    vocabulary = read(data_dir / 'vocabulary.json')
    n_entities, n_relations = len(vocabulary['entities']), len(vocabulary['relations'])
    train_ids = triples[split == 0]
    measurable = trained_nodes[triples[:, 0]] & trained_nodes[triples[:, 2]] & trained_relations[triples[:, 1]]
    known = np.sort(triple_keys(triples, n_entities, n_relations))
    negative = lambda positive, rng: sample_typed_negatives(positive, domains, trained_nodes, known, n_relations, rng)
    evaluation = {}
    # Fixed evaluation candidates, frozen before the first optimizer step.
    for fold, label in ((1, 'validation'), (2, 'test')):
        rows = triples[(split == fold) & measurable]
        rng = np.random.default_rng(9000 + fold)
        questions = rows[np.sort(rng.choice(len(rows), min(CONFIG['ranking_queries'], len(rows)), replace=False))]
        rank_rows = np.concatenate((questions[:, None, :],
                                   negative(np.repeat(questions, CONFIG['ranking_contrasts'], axis=0), rng)
                                   .reshape(len(questions), CONFIG['ranking_contrasts'], 3)), axis=1)
        evaluation[label] = dict(positive=rows, negative=negative(np.repeat(rows, 5, axis=0), rng), rank_rows=rank_rows)
    evaluation_arrays = {f'{fold}_{name}': value for fold,packet in evaluation.items() for name,value in packet.items()}
    if (run/'EVALUATION_CANDIDATES.npz').exists():
        with np.load(run/'EVALUATION_CANDIDATES.npz',allow_pickle=False) as previous:
            if set(previous.files)!=set(evaluation_arrays) or any(not np.array_equal(previous[k],v) for k,v in evaluation_arrays.items()):
                raise ValueError('Frozen evaluation candidate drift')
    else:
        with (run / 'EVALUATION_CANDIDATES.npz').open('xb') as stream:
            np.savez(stream, **evaluation_arrays)
    progress('allocate_model', train_triples=len(train_ids), entities=n_entities, relations=n_relations)
    blocks = relation_blocks(train_ids, n_entities, n_relations, device)
    model = GNNLinkPredictor(n_entities,n_relations, **{k:CONFIG[k] for k in
                             ('model','embedding_dim','layers','dropout','decoder','reverse_relations')}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    rng = np.random.default_rng(CONFIG['seed'])
    if CONFIG != auth['config']:
        raise ValueError('Bound configuration differs from trainer')

    @torch.no_grad()
    def predict(z, rows):
        chunks = []
        for i in range(0, len(rows), CONFIG['decoder_batch_size']):
            values = model.score(z, torch.as_tensor(rows[i:i+CONFIG['decoder_batch_size']], device=device))
            chunks.append(values.cpu().numpy())
        result = np.concatenate(chunks)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite prediction')
        return result

    def evaluate(z, label):
        packet = evaluation[label]
        positive = predict(z, packet['positive'])
        corruptions = predict(z, packet['negative'])
        labels = np.r_[np.ones(len(positive)),np.zeros(len(corruptions))]
        scores = np.r_[positive, corruptions]
        result = ranking(predict(z, packet['rank_rows'].reshape(-1,3)).reshape(len(packet['rank_rows']),-1))
        result.update(auroc=float(roc_auc_score(labels,scores)),
                      average_precision=float(average_precision_score(labels,scores)),
                      positive_count=len(positive), contrast_count=len(corruptions),
                      total_fold_triples=manifest['folds']['1' if label=='validation' else '2']['total'],
                      scientific_accuracy=None)
        return result

    def checkpoint_payload(epoch):
        return dict(state_dict=cpu_tree(model.state_dict()), config=CONFIG,
                    graph_revision=manifest['graph_revision'], dataset_sha256=auth['dataset_sha256'],
                    execution_sha256=auth.get('trajectory_execution_sha256', auth['execution_sha256']), epoch=epoch)

    best, best_epoch, stale = -1., 0, 0
    history, next_epoch, selected_state = [], 1, None
    if auth.get('resume_from'):
        binding = auth['resume_from']
        parent = Path(binding['run'])
        if sha(parent/'resume.pt') != binding['files']['resume.pt']:
            raise ValueError('Parent checkpoint changed before recovery')
        payload, selected_state, history, _ = load_committed(parent)
        next_epoch, best, best_epoch, stale = restore_training_state(payload, model, optimizer, rng)
        if stop_reason(payload):
            # Last committed epoch can be recovered just to finish its export.
            next_epoch = CONFIG['epochs']+1
        del payload
    progress('training_ready')
    for epoch in range(next_epoch,CONFIG['epochs']+1):
        begun = time.monotonic()
        progress('epoch_encoder', epoch=epoch)
        loss_sum, grad, n_contrasts = optimize_epoch(model, optimizer, blocks, train_ids, domains,
                                                    trained_nodes, known, rng, device, CONFIG, check)
        model.eval()
        with torch.no_grad():
            z = encode_sparse(model,blocks)
            validation = evaluate(z,'validation')
        del z
        improved = validation['sampled_mrr'] > best + 1e-7
        if improved:
            best,best_epoch,stale = validation['sampled_mrr'],epoch,0
            selected_state = checkpoint_payload(epoch)
            save_checkpoint(run/'selected.pt',selected_state)
        else:
            stale += 1
        record = dict(epoch=epoch, loss=loss_sum, validation=validation, improved=improved,
                      positive_count=len(train_ids),contrast_count=n_contrasts,
                      best_epoch=best_epoch,gradient_norm_before_clip=grad,seconds=time.monotonic()-begun,
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        # Commit the record and selected weights with optimizer/RNG atomically.
        # Recovery can reconstruct a missing last journal row or an overwritten
        # selected.pt from this commit, without repeating an optimizer step.
        resume = checkpoint_payload(epoch)
        resume.update(optimizer=cpu_tree(optimizer.state_dict()), rng_numpy=rng.bit_generator.state,
                      rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                      rng_python=random.getstate(), best=best,best_epoch=best_epoch,stale=stale,
                      epoch_record=record, selected_checkpoint=selected_state)
        save_checkpoint(run/'resume.pt',resume)
        del resume
        write(run/'epochs'/f'{epoch:03}.json',record)
        history.append(record)
        progress('epoch_complete', **record,
                 remaining_50_epoch_estimate_seconds=(CONFIG['epochs']-epoch)*float(np.median([r['seconds'] for r in history[-3:]])))
        if epoch >= CONFIG['minimum_epochs'] and stale >= CONFIG['patience']:
            break
    selected = torch.load(run/'selected.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(selected['state_dict'])
    model.eval()
    progress('final_test',selected_epoch=best_epoch)
    with torch.no_grad():
        z = encode_sparse(model,blocks)
        test = evaluate(z,'test')
        with (run/'entity_embeddings.npy').open('xb') as stream:
            np.save(stream,z.cpu().numpy(),allow_pickle=False)
    degrees = np.bincount(train_ids[:,[0,2]].ravel(),minlength=n_entities)
    questions = evaluation['test']['rank_rows']
    baseline = ranking(np.log1p(degrees[questions[:,:,0]])*np.log1p(degrees[questions[:,:,2]]))
    with (run/'exposure.npz').open('xb') as stream:
        np.savez(stream,trained_nodes=trained_nodes,trained_relations=trained_relations)
    if code_binding()!=auth['code']:
        raise ValueError('Training code changed during run')
    if any(fingerprint(p)!=value for p,value in manifest['source_fingerprint'].items()):
        raise ValueError('Source changed during run')
    result = dict(status='TRAINED_PENDING_INDEPENDENT_VERIFICATION',architecture='rgcn',decoder='directional',
                  config=CONFIG,graph_revision=manifest['graph_revision'],dataset_sha256=auth['dataset_sha256'],
                  selected_epoch=best_epoch,epochs_run=len(history),validation_selected_mrr=best,
                  test=test,degree_baseline=baseline,gnn_minus_degree_sampled_mrr=test['sampled_mrr']-baseline['sampled_mrr'],
                  worker_seconds=time.monotonic()-started,peak_cpu_bytes=peak_cpu,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                  evaluation_candidate_sha256=sha(run/'EVALUATION_CANDIDATES.npz'),
                  scientific_accuracy=None,independent_scientific_validation=False,
                  protected_work_count=manifest['protected_work_count'],graph_modified=False,
                  full_eligible_graph=True,files={name:sha(run/name) for name in
                    ('selected.pt','entity_embeddings.npy','exposure.npz')})
    write(run/'RESULT.json',result)
    write(run/'inference_manifest.json',dict(schema='frozen-rgcn-export.v1',architecture='rgcn',decoder='directional',
          graph_revision=manifest['graph_revision'],dataset_sha256=auth['dataset_sha256'],selected_epoch=best_epoch,
          execution_sha256=selected['execution_sha256'],files={**result['files'],'RESULT.json':sha(run/'RESULT.json'),
          'data/vocabulary.json':sha(data_dir/'vocabulary.json'),'data/DATASET.json':sha(data_dir/'DATASET.json')}))
    progress('trained',selected_epoch=best_epoch,test=test)


def launch(run,authorization='AUTHORIZATION.json', *, resume=False):
    run=Path(run).resolve()
    if not run.is_relative_to(ROOT/'tmp'):
        raise ValueError('Training artifacts must stay in workspace tmp')
    auth=read(run/authorization)
    if auth['code'] != code_binding() or CONFIG != auth['config']:
        raise ValueError('Current code/config requires its own bound authorization')
    if list((run/'epochs').glob('*')) or (run/'selected.pt').exists() or (run/'resume.pt').exists():
        raise ValueError('Existing optimizer trajectory requires explicit resume, never restart')
    if bool(auth.get('resume_from')) != resume:
        raise ValueError('Use the explicit resume command for a bound recovery segment')
    if resume:
        prepare_resume(run, auth)
    else:
        (run/'epochs').mkdir(exist_ok=True)
    guard=RunGuard(Path(auth['ledger_path']),auth)
    attempt=guard.reserve(auth['execution_sha256'],auth['max_run_seconds'])
    started=time.monotonic()
    suffix='' if authorization=='AUTHORIZATION.json' else '_'+Path(authorization).stem
    with (run/f'TRAINING{suffix}.log').open('x',encoding='utf-8') as log:
        worker=psutil.Popen([sys.executable,'-B','-u','-m',__name__ if __name__!='__main__' else
                            'models.kg_link_prediction.full_graph_train','worker','--run',str(run),'--attempt',str(attempt),
                            '--authorization',authorization],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(run/f'PROCESS{suffix}.json',dict(supervisor_pid=os.getpid(),worker_pid=worker.pid,attempt=attempt,started_unix=time.time()))
        status=guard.supervise(worker,attempt,started=started)
    write(run/f'WORKER_EXIT{suffix}.json',dict(exit_code=status,seconds=time.monotonic()-started,attempt=attempt))
    if status:
        raise SystemExit(status)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['launch','resume','worker'])
    parser.add_argument('--run',required=True)
    parser.add_argument('--attempt',type=int)
    parser.add_argument('--authorization',default='AUTHORIZATION.json')
    args=parser.parse_args()
    if args.command in {'launch','resume'}:
        launch(args.run,args.authorization,resume=args.command=='resume')
    else:
        train(Path(args.run),args.attempt,args.authorization)
