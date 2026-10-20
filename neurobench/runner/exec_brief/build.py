"""Materialize new input copies; preserve every old packet and trial byte."""
from pathlib import Path
import argparse
import shutil

from neurobench.runner.exec_dev.common import file_map, read_json, sha, write_json

REVISION = 'exec-brief-dev-20261019-r1'
ANALYSIS = '''import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description="Dataset summary statistics")
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
p.add_argument('--nonce',required=True)
a=p.parse_args()
x=np.load(a.input,allow_pickle=False)
r={'qualified':True,'nonce':a.nonce,'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
   'count':int(x.size),'mean':float(x.mean()),'std':float(x.std(ddof=0)),'pid':os.getpid()}
Path(a.output).write_text(json.dumps(r,allow_nan=False)+'\\n',encoding='utf-8')
print('QC_COMPLETE')
'''


def build(destination, repo):
    destination, repo = Path(destination).resolve(), Path(repo).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    draft = repo / 'tmp/benchmark_task_briefs_20261019_r1'
    review = read_json(draft / 'private/REVIEW.json')
    protected = read_json(draft / 'private/BEFORE.json')['files']
    assert all(sha(p) == h for p, h in protected.items()), 'old_binding_changed'
    cases, lineage = [], []
    for item in review['cases']:
        cid = item['draft_case_id']
        old = Path(item['source_contract_path']).parent
        source = read_json(old / 'CONTRACT.json')
        target = destination / 'public/cases' / cid
        (target / 'inputs').mkdir(parents=True)
        for original, alias in item['planned_input_aliases'].items():
            shutil.copy2(old / 'inputs' / original, target / 'inputs' / alias)
        if cid == 'N08':
            p = target / 'inputs/preprocessing.json'
            state = read_json(p)
            del state['injected_fault']
            write_json(p, state)
        if cid in ('N09', 'N10'):
            (target / 'inputs/analysis.py').write_text(ANALYSIS, encoding='utf-8')
        # Retain the actual historical mismatch evidence, without a fault label.
        if cid == 'N10':
            state = read_json(target / 'inputs/stage.json')
            state['nonce'] = 'prior_project_run'
            write_json(target / 'inputs/stage.json', state)
        text = (draft / item['draft_task_path']).read_text(encoding='utf-8')
        (target / 'task.md').write_text(text, encoding='utf-8')
        contract = {'revision': REVISION, 'case_id': cid,
                    'title': text.splitlines()[0].lstrip('# '),
                    'required_outputs': item['nominal_requested_outputs'],
                    'input_sha256': file_map(target / 'inputs'), 'absolute_tolerance': 1e-6}
        # Schemas specify products, without diagnostic/solution annotations.
        if cid == 'N08':
            contract['output_fields'] = {'scaler.json': ['mean', 'scale', 'fit_rows', 'statistical_conventions']}
        if cid == 'N07':
            contract['output_fields'] = {'design.json': ['seed', 'method', 'rounding']}
        write_json(target / 'CONTRACT.json', contract)
        cases.append({'case_id': cid, 'contract_sha256': sha(target / 'CONTRACT.json'),
                      'task_sha256': sha(target / 'task.md')})
        lineage.append({'case_id': cid, 'source_case': item['source_case_id'],
                        'family_id': source['family_id'], 'input_kind': source['input_kind'],
                        'source_contract_sha256': sha(old / 'CONTRACT.json'),
                        'aliases': item['planned_input_aliases']})
    write_json(destination / 'INDEX.json', {'revision': REVISION, 'cases': cases})
    write_json(destination / 'PRIVATE_LINEAGE.json', {'cases': lineage, 'not_model_visible': True})
    write_json(destination / 'PRESERVED.json', {'files': protected, 'checked': len(protected)})
    print({'built': str(destination), 'cases': len(cases), 'preserved': len(protected)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--destination', required=True, type=Path)
    p.add_argument('--repo', default=Path.cwd(), type=Path)
    a = p.parse_args()
    build(a.destination, a.repo)
