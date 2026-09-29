"""Verify native-demo ZIP, executable checks, and packaged synthetic arrays."""
import hashlib
import json
import csv
import io
import re
from pathlib import Path
import zipfile
from importlib.util import spec_from_file_location, module_from_spec

desktop = Path(__file__).resolve().parents[1]
dist = desktop / 'dist-demo-native'
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

archive = dist / 'NeuroDiscovery-Native-Demo-1.0.0-win-x64.zip'
portable = dist / 'NeuroDiscovery-Native-Demo-1.0.0-Portable-x64.exe'
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    names = set(bundle.namelist())
    assert 'NeuroDiscovery.exe' in names
    assert not any('runtime/python' in name or name.endswith('/.env') for name in names)
    assert hashlib.sha256(bundle.read('resources/app.asar')).hexdigest() == sha(dist / 'win-unpacked/resources/app.asar')
    guide = bundle.read('resources/DEMO_README.md')
    assert guide == (desktop / 'DEMO_PROMPTS.md').read_bytes()
    assert not re.search(r'[\u3400-\u9fff]', guide.decode('utf-8'))
    manifest = json.loads(bundle.read('resources/demo-workspace/data/demo/MANIFEST.json'))
    for item in manifest['files']:
        assert hashlib.sha256(bundle.read('resources/demo-workspace/data/demo/' + item['path'])).hexdigest() == item['sha256']
    pool = list(csv.DictReader(io.StringIO(bundle.read('resources/demo-workspace/ideas/adhd_network/hypotheses.csv').decode('utf-8-sig'))))
    assert len(pool) == 500
    assert len({row['id'] for row in pool}) == len({row['hypothesis'] for row in pool}) == 500
    reviewed = [row for row in pool if row['review_status'] == 'reviewed']
    assert len(reviewed) == 8
    scores = [float(row['total']) for row in reviewed]
    assert scores == sorted(scores, reverse=True)
    assert all(row['data_kind'] == 'simulated' for row in pool)
    assert not re.search(r'[\u3400-\u9fff]', json.dumps(pool, ensure_ascii=False))
    assert all(not row[field] for row in pool if row['review_status'] == 'not_reviewed'
               for field in ['novelty', 'statistical', 'clinical', 'methodological', 'review_mean', 'total'])
spec = spec_from_file_location('verify_samples', desktop / 'tests/verify-showcase-data.py')
verify_samples = module_from_spec(spec)
spec.loader.exec_module(verify_samples)
data = verify_samples.verify(dist / 'win-unpacked/resources/demo-workspace/data/demo')
checks = desktop.parent / 'tmp/demo-native-check'
receipts = {name: json.loads((checks / name).read_text(encoding='utf-8')) for name in ['RESULTS.json', 'PACKAGED.json', 'PORTABLE.json']}
assert all(receipt['ok'] for receipt in receipts.values())
artifact_preview = json.loads((desktop.parent / 'tmp/artifact-panel-check/RESULTS.json').read_text(encoding='utf-8'))
assert artifact_preview['ok']
presentation = json.loads((desktop.parent / 'tmp/presentation-check/RESULTS.json').read_text(encoding='utf-8'))
assert presentation['ok']
artifacts = [dict(name=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in [archive, portable]]
report = dict(ok=True, distribution='native-ui-demo', artifacts=artifacts, zip_crc=True,
              production_frontend_identical=True, examples_language='en', hypothesis_pool_rows=len(pool), reviewed_preview_rows=len(reviewed), data=data, checks=receipts, artifact_preview=artifact_preview,
              presentation=presentation, model_calls=0, scientific_validation=False)
(dist / 'RELEASE_MANIFEST.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
(dist / 'SHA256SUMS.txt').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in artifacts), encoding='utf-8')
print(json.dumps(dict(ok=True, artifacts=artifacts, hypothesis_pool_rows=len(pool), reviewed_preview_rows=len(reviewed), data=data), ensure_ascii=False))
