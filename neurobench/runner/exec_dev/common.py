"""Serialization, input bindings and bounded paths shared by case consumers.

Expected scientific/numerical answers are deliberately not shared here.
Filesystem checks are cooperative controls, not an operating-system sandbox.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REVISION = 'neuroruntime-exec-dev-20261019-r1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def contained(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or '..' in relative.parts:
        raise ValueError('non_relative_path')
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('path_outside_root')
    return result


def file_map(root):
    root = Path(root)
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction()):
            raise ValueError('linked_file_or_directory')
        if p.is_file():
            rel = str(p.relative_to(root)).replace('\\', '/')
            result[rel] = sha(contained(root, rel))
    return result


def receipt_digest(root):
    return hashlib.sha256(json.dumps(file_map(root), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    root: Path
    contract: dict

    @property
    def id(self):
        return self.contract['case_id']

    def input(self, name):
        return contained(self.root / 'inputs', name)


def load_case(packet, case_id, expected_index_sha=None):
    packet = Path(packet).resolve()
    index_path = packet / 'INDEX.json'
    if expected_index_sha is not None and sha(index_path) != expected_index_sha:
        raise ValueError('packet_index_binding_changed')
    index = read_json(index_path)
    entry = next(x for x in index['cases'] if x['case_id'] == case_id)
    root = contained(packet / 'public/cases', case_id)
    if sha(root / 'CONTRACT.json') != entry['contract_sha256']:
        raise ValueError('contract_binding_changed')
    if sha(root / 'task.md') != entry['task_sha256']:
        raise ValueError('task_binding_changed')
    case = Case(root, read_json(root / 'CONTRACT.json'))
    if file_map(root / 'inputs') != case.contract['input_sha256']:
        raise ValueError('input_binding_changed')
    return case


def finish(case, output, status='complete', **fields):
    output = Path(output)
    artifacts = file_map(output)
    artifacts.pop('result.json', None)
    write_json(output / 'result.json', {
        'case_id': case.id, 'status': status,
        'input_sha256': case.contract['input_sha256'],
        'artifacts': artifacts, **fields,
    })


def environment():
    import numpy
    import nibabel
    import scipy
    return {'python': sys.version, 'executable': sys.executable,
            'prefix': sys.prefix, 'os': os.name, 'numpy': numpy.__version__,
            'nibabel': nibabel.__version__, 'scipy': scipy.__version__}
