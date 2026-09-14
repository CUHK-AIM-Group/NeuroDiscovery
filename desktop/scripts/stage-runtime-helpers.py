"""Copy the explicit runtime import closure, never run a research CLI or copy its data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def stage(source: Path, backend: Path, manifest: Path) -> int:
    source, backend = source.resolve(), backend.resolve()
    names = json.loads(manifest.read_text(encoding='utf-8'))
    if (not isinstance(names, list) or not all(isinstance(name, str) for name in names)
            or len(names) != len(set(names))):
        raise ValueError('Expected a unique runtime helper allowlist')
    pairs = []
    for name in names:
        relative = Path(name)
        if (relative.is_absolute() or len(relative.parts) != 3
                or relative.parts[:2] != ('neurooracle', 'scripts') or relative.suffix != '.py'):
            raise ValueError('Only explicitly named NeuroOracle Python helpers may be staged')
        src, dest = (source / relative).resolve(), (backend / relative).resolve()
        src.relative_to(source)
        dest.relative_to(backend)
        if not src.is_file():
            raise FileNotFoundError(f'Missing runtime helper: {name}')
        if dest.exists() and dest.read_bytes() != src.read_bytes():
            raise ValueError(f'Refusing to replace a different staged helper: {name}')
        pairs.append((src, dest))
    for src, dest in pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(src, dest)
    return len(pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--backend', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'runtime-helper-files.json')
    args = parser.parse_args()
    print(f'Staged {stage(args.source, args.backend, args.manifest)} unchanged runtime import helpers; no campaign was executed.')
