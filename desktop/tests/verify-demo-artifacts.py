import hashlib
import json
import zipfile
from pathlib import Path


DESKTOP = Path(__file__).resolve().parents[1]
OUTPUT = DESKTOP / 'dist-demo'
archive = OUTPUT / 'NeuroDiscovery-Demo-1.0.0-win-x64.zip'
portable = OUTPUT / 'NeuroDiscovery-Demo-1.0.0-Portable-x64.exe'

with zipfile.ZipFile(archive) as package:
    assert package.testzip() is None
    names = package.namelist()
    assert 'NeuroDiscovery Demo.exe' in names
    assert 'resources/runtime/python/python.exe' in names
    assert not any('.frozen/' in name or '__pycache__/' in name for name in names)
    backend = 'resources/runtime/backend/'
    assert json.loads(package.read(backend + 'DEMO_DISTRIBUTION.json'))['human_evaluation'] is False
    for name in names:
        if not name.startswith(backend):
            continue
        relative = name.removeprefix(backend)
        assert not relative.startswith(('core/web/study_materials/', 'neurooracle/data/user_study/'))
        assert Path(relative).name not in {'study.html', 'user_study.py', '.env', 'desktop-config.json'}
        assert not Path(relative).name.startswith(('evaluation', 'discovery-study'))
    page = package.read(backend + 'core/web/static/index.html').decode('utf-8')
    assert 'const HUMAN_EVALUATION_ENABLED = false;' in page
    config = json.loads(package.read(backend + 'neuroclaw_environment.json'))
    assert not config['llm_backend'].get('api_key')
    assert not any(name.endswith(('.sqlite', '.db')) for name in names)

assert portable.read_bytes()[:2] == b'MZ'
artifacts = []
for artifact in (portable, archive):
    with artifact.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    artifacts.append({'file': artifact.name, 'bytes': artifact.stat().st_size, 'sha256': digest})

receipt = {
    'distribution': 'demo',
    'platform': 'windows-x64',
    'human_evaluation': False,
    'checks': ['ZIP CRC', 'demo marker', 'no study assets', 'no frozen research archives',
               'no database files', 'default configuration has no API key', 'UI study views disabled', 'PE header'],
    'artifacts': artifacts,
}
(OUTPUT / 'VERIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
(OUTPUT / 'SHA256SUMS.txt').write_text(''.join(f"{item['sha256']}  {item['file']}\n" for item in artifacts), encoding='utf-8')
print(json.dumps(receipt, indent=2))
