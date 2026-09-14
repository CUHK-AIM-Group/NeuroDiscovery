"""Source-level guards for both distribution staging entry points."""
import json
import importlib.util
import re
from pathlib import Path

import pytest

DESKTOP = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('platform', ['win', 'mac'])
def test_reusable_models_are_staged_without_weights_or_local_experiments(platform):
    name = 'prepare-bundled-runtime.ps1' if platform == 'win' else 'prepare-bundled-runtime-mac.sh'
    script = (DESKTOP / 'scripts' / name).read_text(encoding='utf-8')
    if platform == 'win':
        assert '@("core", "skills", "neurooracle", "models")' in script
    else:
        assert 'stage_backend_dir "models"' in script
    for excluded in ('checkpoints', 'benchmark_results', 'experiment_results',
                     '*.pt', '*.pth', '*.ckpt', '*.safetensors',
                     'sweep_atlases.py', 'sweep_targets.py', 'tune_braingnn.py', 'run_benchmark.py'):
        assert excluded in script


def test_desktop_and_lockfile_versions_match():
    package = json.loads((DESKTOP / 'package.json').read_text(encoding='utf-8'))
    lock = json.loads((DESKTOP / 'package-lock.json').read_text(encoding='utf-8'))
    assert package['version'] == lock['version'] == lock['packages']['']['version']


@pytest.mark.parametrize('filename,constant', [
    ('main.js', 'BUNDLED_RUNTIME_VERSION'),
    ('preload.js', 'DESKTOP_VERSION'),
])
def test_runtime_and_display_versions_match_package(filename, constant):
    version = json.loads((DESKTOP / 'package.json').read_text(encoding='utf-8'))['version']
    source = (DESKTOP / filename).read_text(encoding='utf-8')
    match = re.search(rf"const {constant} = '([^']+)';", source)
    assert match and match.group(1) == version


def test_backend_client_identification_matches_package_version():
    version = json.loads((DESKTOP / 'package.json').read_text(encoding='utf-8'))['version']
    source = (DESKTOP.parent / 'core/web/server.py').read_text(encoding='utf-8')
    assert f'"User-Agent": "NeuroDiscovery/{version}"' in source


def test_packaged_runtime_remains_separate_from_personal_configuration():
    package = json.loads((DESKTOP / 'package.json').read_text(encoding='utf-8'))
    assert package['build']['files'] == ['main.js', 'llm-settings.js', 'llm-credentials.js', 'model-library.js', 'settings-restart.js', 'preload.js', 'package.json']
    assert package['build']['extraResources'][0]['from'] == 'runtime'
    script = (DESKTOP / 'scripts' / 'prepare-bundled-runtime.ps1').read_text(encoding='utf-8')
    assert '$defaultEnvironment = [ordered]@{' in script
    assert 'Copy-RootFileIfExists -SourceRoot $RepoRoot -TargetRoot $BackendTarget -Name $fileName' in script
    assert 'foreach ($fileName in @("LICENSE", "README.md", "README_zh.md", "SOUL.md", "pyproject.toml"))' in script


@pytest.mark.parametrize('platform', ['win', 'mac'])
def test_import_time_policy_asset_is_explicitly_staged(platform):
    name = 'prepare-bundled-runtime.ps1' if platform == 'win' else 'prepare-bundled-runtime-mac.sh'
    script = (DESKTOP / 'scripts' / name).read_text(encoding='utf-8').replace('\\', '/')
    assert 'neurooracle/data/case_study_reaudit/full_graph_v3/RUBRIC.md' in script


@pytest.fixture
def staging(tmp_path):
    spec = importlib.util.spec_from_file_location('stage_runtime_helpers', DESKTOP / 'scripts' / 'stage-runtime-helpers.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, backend = tmp_path / 'source', tmp_path / 'backend'
    (source / 'neurooracle/scripts').mkdir(parents=True)
    backend.mkdir()
    manifest = tmp_path / 'manifest.json'
    return module.stage, source, backend, manifest


def test_runtime_helper_staging_is_exact_nonexecuting_and_idempotent(staging):
    stage, source, backend, manifest = staging
    name = 'neurooracle/scripts/helper.py'
    payload = b'raise RuntimeError("This module must only be copied, never executed")\n'
    (source / name).write_bytes(payload)
    (source / 'neurooracle/scripts/private.json').write_text('not a runtime input')
    manifest.write_text(json.dumps([name]))
    assert stage(source, backend, manifest) == 1
    assert stage(source, backend, manifest) == 1
    assert (backend / name).read_bytes() == payload
    assert not (backend / 'neurooracle/scripts/private.json').exists()


@pytest.mark.parametrize('names', [
    {'helper': 'neurooracle/scripts/helper.py'},
    [None],
    ['neurooracle/scripts/helper.py', 'neurooracle/scripts/helper.py'],
    ['neurooracle/scripts/../../private.py'],
    ['neurooracle/scripts/nested/helper.py'],
    ['core/scripts/helper.py'],
    ['neurooracle/scripts/private.json'],
])
def test_runtime_helper_allowlist_rejects_invalid_targets(staging, names):
    stage, source, backend, manifest = staging
    manifest.write_text(json.dumps(names))
    with pytest.raises(ValueError):
        stage(source, backend, manifest)
    assert not list(backend.rglob('*'))


def test_runtime_helper_staging_validates_all_inputs_before_copying(staging):
    stage, source, backend, manifest = staging
    name = 'neurooracle/scripts/helper.py'
    (source / name).write_text('present')
    manifest.write_text(json.dumps([name, 'neurooracle/scripts/missing.py']))
    with pytest.raises(FileNotFoundError):
        stage(source, backend, manifest)
    assert not list(backend.rglob('*'))


def test_runtime_helper_staging_preserves_different_existing_files(staging):
    stage, source, backend, manifest = staging
    name = 'neurooracle/scripts/helper.py'
    (source / name).write_text('source')
    (backend / name).parent.mkdir(parents=True)
    (backend / name).write_text('existing')
    manifest.write_text(json.dumps([name]))
    with pytest.raises(ValueError, match='Refusing to replace'):
        stage(source, backend, manifest)
    assert (backend / name).read_text() == 'existing'
