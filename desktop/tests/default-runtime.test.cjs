const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../main.js'), 'utf8');
const functions = ['defaultConfig', 'loadConfig', 'resolveRuntimeConfig'].map(name =>
  source.match(new RegExp('^function ' + name + '\\([^\\n]*\\)[\\s\\S]+?^}', 'm'))[0]).join('\n');

function context(saved = {}) {
  const sandbox = {
    app: { isPackaged: false, getPath: () => '/user' },
    process: { env: {}, argv: [], platform: 'win32' },
    os: { homedir: () => '/home' }, path,
    fs: { readFileSync: () => JSON.stringify(saved) },
    userConfigPath: () => '/user/settings.json',
    packagedRuntimeSourceRoot: () => '/desktop/runtime',
    bundledPythonExe: root => `${root || '/cached'}/python/python.exe`,
    bundledBackendRoot: () => '/cached/backend',
    repoRoot: () => '/checkout', defaultCondaExe: () => '/conda',
    defaultLlmBaseUrl: () => 'https://example.invalid',
    normalizeConfig: value => value, normalizePackagedRuntimeConfig: value => value,
    ensureBundledRuntime: () => ({ pythonExe: '/desktop/runtime/python/python.exe', repoRoot: '/checkout' }),
  };
  vm.runInNewContext(functions, sandbox);
  return sandbox;
}

test('fresh desktop defaults to built-in Python', () => {
  const runtime = context();
  const config = runtime.defaultConfig();
  assert.equal(config.runtimeMode, 'bundled');
  assert.equal(config.pythonExe, '/desktop/runtime/python/python.exe');
});

test('development launch no longer forces the incomplete venv', () => {
  const runtime = context({ runtimeMode: 'python', pythonExe: '/checkout/.venv/python.exe', localPythonExe: '/other/python' });
  runtime.process.argv.push('--development-runtime');
  const config = runtime.resolveRuntimeConfig(runtime.loadConfig());
  assert.equal(config.runtimeMode, 'bundled');
  assert.equal(config.pythonExe, '/desktop/runtime/python/python.exe');
  assert.equal(config.repoRoot, '/checkout');
  assert.ok(config.environmentFile.endsWith('development-environment.json'));
});

test('ordinary launch preserves explicit alternative runtime selection', () => {
  const runtime = context({ runtimeMode: 'python', localPythonExe: '/custom/python' });
  assert.equal(runtime.resolveRuntimeConfig(runtime.loadConfig()).pythonExe, '/custom/python');
});
