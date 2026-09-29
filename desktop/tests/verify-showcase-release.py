"""Check the produced Windows ZIP and bind release artifacts to tested sources."""
import hashlib
import json
from pathlib import Path
import zipfile

DESKTOP = Path(__file__).resolve().parents[1]
DIST = DESKTOP / 'dist-demo-showcase'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


if __name__ == '__main__':
    archive = DIST / 'NeuroDiscovery-Offline-Demo-1.0.0-win-x64.zip'
    executable = DIST / 'NeuroDiscovery-Offline-Demo-1.0.0-Portable-x64.exe'
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        names = set(bundle.namelist())
        assert 'NeuroDiscovery Demo.exe' in names
        assert 'resources/app.asar' in names
        assert 'resources/demo-workspace/data/demo/MANIFEST.json' in names
        assert not any('runtime/python' in name or name.endswith('/.env') for name in names)
        assert hashlib.sha256(bundle.read('resources/app.asar')).hexdigest() == sha(DIST / 'win-unpacked/resources/app.asar')
        manifest = json.loads(bundle.read('resources/demo-workspace/data/demo/MANIFEST.json'))
        for item in manifest['files']:
            data = bundle.read('resources/demo-workspace/data/demo/' + item['path'])
            assert hashlib.sha256(data).hexdigest() == item['sha256']
    checks = DESKTOP.parent / 'tmp/demo-showcase-check'
    receipts = {name: json.loads((checks / name).read_text()) for name in ['RESULTS.json', 'PACKAGED.json', 'PORTABLE.json']}
    assert all(receipt['ok'] for receipt in receipts.values())
    artifacts = [{"name": file.name, "bytes": file.stat().st_size, "sha256": sha(file)} for file in [executable, archive]]
    report = dict(distribution='offline-demo', version='1.0.0', platform='Windows x64',
                  artifacts=artifacts, zip_crc=True, zip_matches_tested_asar=True,
                  packaged_fixture_hashes=True, source_ui_checks=receipts['RESULTS.json'],
                  packaged_launch=True, portable_launch=True, scientific_validation=False,
                  data_kind='synthetic', llm_calls=0, real_training=False)
    (DIST / 'RELEASE_MANIFEST.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (DIST / 'SHA256SUMS.txt').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in artifacts), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
