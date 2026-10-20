from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from neurobench.runner.exec_dev.common import (
    Case, contained, file_map, load_case, read_csv, read_json, sha, write_csv,
    write_json,
)

ROOT = Path(os.environ.get('NEUROCLAW_HARD_REPO') or Path(__file__).resolve().parents[3]).resolve()
CASE_IDS = ('H01', 'H02', 'H03', 'H05', 'H07', 'H09', 'H10', 'H11')
MODELS = ('gpt-6.1-sol', 'deepseek-v4.1-flash')
ARMS = ('NR_OFF', 'NR_ON')
REVISION = 'hard-execution-development-20261020-r2'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def arrays_close(a, b, atol=5e-6, rtol=5e-5):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.isfinite(a).all()) and bool(
        np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))


def finalize(case, output):
    from neurobench.runner.exec_dev.common import finish
    finish(case, output)


def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
