"""Separate-process checks, including an independent NumPy two-layer encoder."""
from __future__ import annotations

import pyarrow  # noqa: F401
import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .full_graph_data import ROOT, fingerprint, read, sha, write


def numpy_encode_subset(state, triples, desired, relations):
    """Independently compute two R-GCN layers for selected nodes, in float64."""
    initial = state['entity_embedding.weight'].numpy()
    desired = np.unique(np.asarray(desired,dtype=np.int64))
    mask = np.zeros(len(initial),dtype=bool)
    mask[desired] = True
    first_nodes = np.unique(np.r_[desired,triples[mask[triples[:,2]],0],triples[mask[triples[:,0]],2]])

    def layer(nodes, inputs, input_ids, number):
        lookup = (lambda ids:inputs[ids]) if input_ids is None else (lambda ids:inputs[np.searchsorted(input_ids,ids)])
        root = state[f'layers.{number}.root'].numpy().astype(np.float64)
        weights = state[f'layers.{number}.weight'].numpy().astype(np.float64)
        bias = state[f'layers.{number}.bias'].numpy().astype(np.float64)
        values = lookup(nodes).astype(np.float64) @ root
        selected = np.zeros(len(initial),dtype=bool)
        selected[nodes] = True
        for reverse in (False,True):
            source,target=(triples[:,2],triples[:,0]) if reverse else (triples[:,0],triples[:,2])
            keep = selected[target]
            src,dst,rel=source[keep],target[keep],triples[keep,1]
            for r in np.unique(rel):
                choose=rel==r
                local=np.searchsorted(nodes,dst[choose])
                sums=np.zeros((len(nodes),initial.shape[1]),dtype=np.float64)
                np.add.at(sums,local,lookup(src[choose]).astype(np.float64))
                counts=np.bincount(local,minlength=len(nodes))
                active=counts>0
                sums[active]/=counts[active,None]
                values[active]+=sums[active]@weights[int(r)+(relations if reverse else 0)]
        return np.maximum(values+bias,0)

    first=layer(first_nodes,initial,None,0)
    final=layer(desired,first,first_nodes,1)
    return desired,final,len(first_nodes)


def independent_logits(embeddings,matrices,bias,rows):
    out=np.empty(len(rows),dtype=np.float64)
    for relation in np.unique(rows[:,1]):
        ids=np.flatnonzero(rows[:,1]==relation)
        source=np.asarray(embeddings[rows[ids,0]],dtype=np.float64)
        target=np.asarray(embeddings[rows[ids,2]],dtype=np.float64)
        out[ids]=np.sum((source@matrices[int(relation)])*target,axis=1)+bias[int(relation)]
    return out


def verify(run, output=None):
    run=Path(run).resolve()
    output = run if output is None else Path(output).resolve()
    if output != run:
        output.mkdir(parents=True, exist_ok=False)
    started=time.monotonic()
    result=read(run/'RESULT.json')
    dataset=read(run/'data/DATASET.json')
    auth=read(run/('AUTHORIZATION_REPAIR1.json' if (run/'AUTHORIZATION_REPAIR1.json').exists() else 'AUTHORIZATION.json'))
    with sqlite3.connect(auth['ledger_path']) as db:
        attempts=db.execute('SELECT state,elapsed FROM attempts').fetchall()
    if len(attempts)!=1 or attempts[0][0]!='COMPLETE':
        raise ValueError('Worker/accounting not complete')
    for relative,digest in auth['code'].items():
        # Verify the exact executed source snapshot, not a later corrected
        # working tree. Both the old and new code remain available for review.
        snapshot = run/'source_snapshot'/Path(relative).name
        source = snapshot if snapshot.exists() else ROOT/relative
        if sha(source)!=digest:
            raise ValueError('Frozen training source snapshot drift')
    epochs=[read(p) for p in sorted((run/'epochs').glob('*.json'))]
    if [x['epoch'] for x in epochs]!=list(range(1,result['epochs_run']+1)):
        raise ValueError('Missing/duplicate optimizer epochs')
    best=max(epochs,key=lambda x:(x['validation']['sampled_mrr'],-x['epoch']))
    if best['epoch']!=result['selected_epoch']:
        raise ValueError('Checkpoint did not follow validation selection')
    if any(x['positive_count']!=dataset['folds']['0']['total'] for x in epochs):
        raise ValueError('Not all training positives used each epoch')
    if result['peak_gpu_reserved_bytes']>auth['max_gpu_bytes'] or result['peak_cpu_bytes']>auth['max_cpu_bytes']:
        raise ValueError('Resource cap exceeded')
    for path,before in dataset['source_fingerprint'].items():
        if fingerprint(path)!=before:
            raise ValueError(f'Input changed: {path}')
    if sha(dataset['graph'])!=dataset['graph_revision']:
        raise ValueError('Graph content hash changed')
    for name,digest in dataset['files'].items():
        if sha(run/'data'/name)!=digest:
            raise ValueError('Training data content changed')
    if sha(run/'EVALUATION_CANDIDATES.npz') != result['evaluation_candidate_sha256']:
        raise ValueError('Pretraining evaluation candidates changed')
    write(output/'idea_gnn.json',dict(kind='rgcn_export',graph_revision=dataset['graph_revision'],
          training_scope='Full eligible current-graph relation reconstruction; source/version and endpoint-pair separated folds; '
                         'training-only message graph; untrained vocabulary excluded; sampled typed-corruption evaluation, '
                         'not calibrated truth, literature novelty or whole-chain scientific validation.',
          export=dict(path=str(run/'inference_manifest.json'),sha256=sha(run/'inference_manifest.json'))))
    from core.idea_ranking import configured_scorer
    from neurooracle.src.kge.hypothesis_path import score_candidate
    os.environ['NEUROCLAW_IDEA_GNN_MANIFEST']=str(output/'idea_gnn.json')
    scorer,receipt=configured_scorer(SimpleNamespace(),dict(graph_revision=dataset['graph_revision']))
    if scorer is None:
        raise ValueError(receipt)
    checkpoint=torch.load(run/'selected.pt',map_location='cpu',weights_only=True)
    state=checkpoint['state_dict']
    resume=torch.load(run/'resume.pt',map_location='cpu',weights_only=True)
    steps=[int(s['step'].item()) for s in resume['optimizer']['state'].values()]
    if resume['epoch']!=len(epochs) or not steps or set(steps)!={len(epochs)}:
        raise ValueError('Resume optimizer step counters disagree with completed epochs')
    if resume['execution_sha256']!=checkpoint['execution_sha256']:
        raise ValueError('Selected and resumable checkpoints belong to different runs')
    resume_sha=sha(run/'resume.pt')
    del resume
    matrices=state['decoder.bilinear'].numpy().astype(np.float64)
    bias=state['decoder.bias'].numpy().astype(np.float64)
    data=np.load(run/'data/data.npz',allow_pickle=False)
    triples,fold=data['triples'],data['split']
    training=triples[fold==0]
    pools=[]
    desired=[]
    old=read(ROOT/'tmp/idea_gnn_reuse_20260929/verified/RESULTS.json')
    for name,old_case in zip(('mci','adhd','genetics'),old['topics']):
        path=ROOT/'tmp/idea_chain_pool_20260929/verified'/f'{name}.json'
        if sha(path)!=old_case['source_pool_sha256']:
            raise ValueError('Previously checked candidate pool changed')
        pool=read(path)
        rows=[]
        for candidate in pool['candidates']:
            scores=score_candidate(candidate,scorer=scorer)
            actual=scores['gnn_path_score']
            triples_for_candidate=candidate['kg_triples']
            measurable=all(t['source'] in scorer.ent2idx and t['target'] in scorer.ent2idx and t['relation'] in scorer.rel2idx for t in triples_for_candidate)
            if actual is not None:
                assert measurable
                encoded=np.array([[scorer.ent2idx[t['source']],scorer.rel2idx[t['relation']],scorer.ent2idx[t['target']]] for t in triples_for_candidate])
                logits=independent_logits(scorer.embeddings,matrices,bias,encoded)
                values=1/(1+np.exp(-np.clip(logits,-700,700)))
                expected=float(np.prod(values)**(1/len(values)))
                if min(values)<.3:
                    expected*=.7
                if not math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9):
                    raise ValueError('Candidate path arithmetic mismatch')
                desired.extend(encoded[:,[0,2]].ravel().tolist())
            elif measurable:
                raise ValueError('Unexpected unmeasured candidate')
            rows.append(dict(candidate_id=candidate['candidate_id'],gnn_score=actual,
                             unresolved=scores['unresolved'],edge_scores=scores['edge_scores']))
        write(output/f'{name}_scores.json',dict(topic=pool['topic'],scores=rows))
        measured=[r['gnn_score'] for r in rows if r['gnn_score'] is not None]
        pools.append(dict(topic=pool['topic'],candidates=len(rows),scored=len(measured),missing=len(rows)-len(measured),
                          old_scored=old_case['scored'],source_pool_sha256=sha(path),
                          score_min=min(measured),score_median=float(np.median(measured)),score_max=max(measured)))
    rng=np.random.default_rng(909)
    known_nodes=np.flatnonzero(data['trained_nodes'])
    desired.extend(rng.choice(known_nodes,128,replace=False).tolist())
    ids,expected,first_count=numpy_encode_subset(state,training,desired,len(matrices))
    observed=np.asarray(scorer.embeddings[ids],dtype=np.float64)
    np.testing.assert_allclose(observed,expected,rtol=2e-4,atol=2e-5)
    encoder_error=float(np.max(np.abs(observed-expected)))
    # Recompute reported test metrics from saved, pre-training candidates, using
    # independently decoded exported embeddings rather than the trainer function.
    ev=np.load(run/'EVALUATION_CANDIDATES.npz',allow_pickle=False)
    p=independent_logits(scorer.embeddings,matrices,bias,ev['test_positive'])
    n=independent_logits(scorer.embeddings,matrices,bias,ev['test_negative'])
    labels=np.r_[np.ones(len(p)),np.zeros(len(n))]
    rank_rows=ev['test_rank_rows']
    scores=independent_logits(scorer.embeddings,matrices,bias,rank_rows.reshape(-1,3)).reshape(len(rank_rows),-1)
    tied=np.isclose(scores[:,1:],scores[:,:1],atol=1e-6,rtol=1e-5)
    ranks=1+((scores[:,1:]>scores[:,:1])&~tied).sum(axis=1)+.5*tied.sum(axis=1)
    metrics=dict(sampled_mrr=float(np.mean(1/ranks)),hits_at_1=float(np.mean(ranks==1)),hits_at_10=float(np.mean(ranks<=10)),
                 average_precision=float(average_precision_score(labels,np.r_[p,n])),auroc=float(roc_auc_score(labels,np.r_[p,n])))
    metric_differences={key:abs(value-result['test'][key]) for key,value in metrics.items()}
    if max(metric_differences.values())>.002:
        raise ValueError(f'Independent metric reproduction failed: {metric_differences}')
    score_receipt=dict(status='INDEPENDENT_NUMERICAL_VERIFICATION_PASSED',
                      result_sha256=sha(run/'RESULT.json'),export_sha256=sha(run/'inference_manifest.json'),
                      graph_sha256_verified=dataset['graph_revision'],input_stats_unchanged=True,
                      executed_source_snapshots_verified=True, verifier_code_sha256=sha(__file__),
                      epoch_log_sha256={p.name:sha(p) for p in (run/'epochs').glob('*.json')},
                      epochs_verified=len(epochs),selected_epoch=best['epoch'],all_training_positives_used=True,
                      resumable_checkpoint_sha256=resume_sha,optimizer_step_counters_verified=steps,
                      independent_numpy_encoder_nodes=len(ids),first_layer_context_nodes=first_count,
                      encoder_max_abs_error=encoder_error,independent_test_metrics=metrics,
                      metric_absolute_differences=metric_differences,pools=pools,
                      scored=sum(x['scored'] for x in pools),candidates=sum(x['candidates'] for x in pools),
                      old_scored=sum(x['old_scored'] for x in pools),runtime_interface=receipt,
                      verification_seconds=time.monotonic()-started,scientific_quality_validated=False,
                      cuda_bitwise_reproducibility_claimed=False,
                      known_limitations=['Graph-derived labels and type-matched corruptions are not scientific true/false labels.',
                         'Test metrics cover only trained-vocabulary rows; report the full denominator and OOV separately.',
                         'Source families use known identity/version records; unknown shared cohorts or aliases remain possible.',
                         'Fixed seeds do not guarantee bitwise CUDA replay; CuBLAS determinism warnings were retained.',
                         'GNN existing-edge compatibility does not prove the whole-chain inference or novelty.'])
    scorer.verify_files()
    write(output/'VERIFIED.json',score_receipt)
    print(json.dumps({k:score_receipt[k] for k in ('status','scored','candidates','old_scored','encoder_max_abs_error','independent_test_metrics','verification_seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',help='New directory for re-verification; keeps the original receipts intact')
    args=parser.parse_args()
    verify(args.run,args.output)
