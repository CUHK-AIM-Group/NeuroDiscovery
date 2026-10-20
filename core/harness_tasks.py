"""Public task and input-header compilation for the shared NeuroRuntime surface."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from core.harness_validation import MAX_TEXT, _task, digest

MAX_INPUTS = 64
MAX_HASH_BYTES = 64 * 1024 * 1024


def _metadata(workspace: Path, relative: str) -> dict:
    root = workspace.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('input_path_outside_workspace')
    if not path.exists():
        return {'exists': False, 'status': 'missing'}
    if not path.is_file():
        return {'exists': True, 'status': 'not_a_file'}
    before = path.stat()
    result = {'exists': True, 'status': 'observed', 'bytes': before.st_size}
    if before.st_size <= MAX_HASH_BYTES:
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            remaining = MAX_HASH_BYTES + 1
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                hasher.update(chunk)
                remaining -= len(chunk)
        result.update(sha256=hasher.hexdigest() if remaining else None,
                      hash_status='computed' if remaining else 'omitted_size_limit')
    else:
        result.update(sha256=None, hash_status='omitted_size_limit')
    try:
        if path.name.endswith('.npy'):
            import numpy as np
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            result.update(shape=list(array.shape), dtype=str(array.dtype))
        elif path.name.endswith(('.nii', '.nii.gz')):
            import nibabel as nib
            image = nib.load(path, mmap=True)
            result.update(shape=list(image.shape), dtype=str(image.get_data_dtype()),
                          affine=image.affine.tolist())
    except (ImportError, OSError, ValueError, EOFError) as error:
        result.update(header_status='unavailable', header_error_type=type(error).__name__,
                      header_error=str(error)[:400])
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        result.update(status='changed_during_read', sha256=None, hash_status='unstable')
    return result


def compile_task(session: Any, arguments: Mapping, workspace: Path, capability) -> dict:
    """Organize supplied public requirements; never supply a workflow or oracle."""
    text = _task(session, arguments)
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise ValueError('task_text_missing_or_too_large')
    contract = arguments.get('public_contract', {})
    if not isinstance(contract, dict) or len(json.dumps(contract, ensure_ascii=False)) > MAX_TEXT:
        raise ValueError('invalid_public_contract')
    paths = arguments.get('input_paths', [])
    if not isinstance(paths, list) or len(paths) > MAX_INPUTS or any(
            not isinstance(p, str) or not p.strip() or len(p) > 4096 for p in paths):
        raise ValueError('invalid_input_paths')
    metadata = {p: _metadata(workspace, p) for p in dict.fromkeys(paths)}
    matches = capability.search(str(contract.get('title') or text), limit=4)
    return {
        'success': True, 'provenance': 'public_task_and_observed_headers',
        'task_text': text, 'task_sha256': digest(text),
        'requirements': contract, 'input_metadata': metadata,
        'method_candidates': [{'name': item.name, 'summary': capability.summary_line(item)} for item in matches],
        'workflow_executed': False, 'scientific_correctness_verified': False,
        'scope': 'Caller-supplied public requirements and workspace file observations; missing inputs remain missing. No processing steps, hidden answers or numerical acceptance criteria are inferred.',
    }
