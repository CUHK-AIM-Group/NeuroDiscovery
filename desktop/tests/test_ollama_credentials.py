"""Synthetic credentials only; no account access or model generation."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_desktop_key_file_is_endpoint_bound_and_never_rotates():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required')
    script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {resolveDesktopApiKey: resolve, keyFileStatus: status, validateKeyFileConfig: validate} = require('./desktop/llm-credentials');
const config = {llmProvider: 'ollama_cloud', llmBaseUrl: 'https://ollama.com/v1',
 llmApiKeyEnv: 'OLLAMA_API_KEY', llmApiKeyFile: path.resolve('synthetic-keys.txt'), llmApiKeySlot: 'primary'};
const primary = 'synthetic-primary-key', reserve = 'synthetic-reserve-key';
const original = fs.readFileSync;
let reads = 0;
fs.readFileSync = () => {reads++; return `unrelated-private-fixture\r\n${primary}\r\n${reserve}\r\n`;};
try {
 assert.equal(resolve(config), primary);
 assert.equal(resolve({...config, llmApiKeySlot: 'reserve'}), reserve);
 const publicStatus = status(config);
 assert.deepEqual(publicStatus, {apiKeyConfigured:true, apiKeySource:'key-file', apiKeySlot:'primary'});
 for (const secret of [primary,reserve]) assert.ok(!JSON.stringify(publicStatus).includes(secret));
 assert.equal(resolve({llmApiKey:'pasted-fixture'}), 'pasted-fixture');
 for (const edit of [
  {llmProvider:'openai'}, {llmBaseUrl:'https://other.example/v1'},
  {llmBaseUrl:'http://ollama.com/v1'}, {llmBaseUrl:'https://ollama.com.evil.example/v1'},
  {llmBaseUrl:'https://ollama.com/v1?redirect=other'}, {llmBaseUrl:'https://user@ollama.com/v1'},
  {llmApiKeySlot:'auto'}, {llmApiKeyFile:'relative.txt'}, {llmApiKey:'pasted-fixture'},
  {llmApiKeyEnv:'OPENAI_API_KEY'},
 ]) {
  const before = reads;
  assert.throws(() => resolve({...config, ...edit}));
  assert.equal(reads, before, 'Invalid routes must be rejected before reading secrets');
 }
 for (const raw of ['', primary, `${primary}\n\n`, `${primary}\n${reserve}\n\n`, `${primary}\ninvalid key`]) {
  fs.readFileSync = () => raw;
  assert.throws(() => resolve(config));
  assert.equal(status(config).apiKeyConfigured, false);
 }
 fs.readFileSync = () => {throw new Error(primary);};
 assert.ok(!JSON.stringify(status(config)).includes(primary));
 assert.throws(() => resolve(config), /Cannot read the Ollama key file/);
} finally {fs.readFileSync = original;}
'''
    subprocess.run([node, '-e', script], cwd=ROOT, check=True, capture_output=True, text=True)


def test_desktop_credentials_packaged_and_not_written_to_backend_config():
    source = (ROOT / 'desktop/main.js').read_text(encoding='utf-8')
    apply_config = source.split('function applyDesktopLlmConfig(config) {', 1)[1].split('function applyLlmProcessEnv', 1)[0]
    assert 'resolveDesktopApiKey' not in apply_config
    assert "config.environmentFile || path.join(config.repoRoot, 'neuroclaw_environment.json')" in apply_config
    assert 'env.NEUROCLAW_ENV_FILE = config.environmentFile' in source
    assert 'llm-credentials.js' in json.loads((ROOT / 'desktop/package.json').read_text())['build']['files']


def test_key_slots_are_visible_only_for_cloud_and_cleared_on_provider_change():
    html = (ROOT / 'core/web/static/index.html').read_text(encoding='utf-8')
    for key in ['llmApiKeyFile', 'llmApiKeySlot']:
        assert f"key: '{key}', scope: 'desktop', whenProvider: 'ollama_cloud'" in html
    assert "state.desktopConfig.llmApiKeyFile = '';" in html
    assert "options: ['primary', 'reserve']" in html
