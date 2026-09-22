const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../evaluation-main.js'), 'utf8');

function fixture(platform = 'darwin', failure = false) {
  const calls = [];
  let complete = false;
  const context = {process: {platform, env: {PYTHONPATH: 'untrusted', PYTHONHOME: 'untrusted'}}, path, require,
    app: {getPath: () => '/user profile'}, fs: {
      readFileSync: () => Buffer.from('{"build":"one"}'), existsSync: () => complete,
      mkdirSync: () => {}, cpSync: (...args) => calls.push(['copy', ...args]),
      writeFileSync: (...args) => {calls.push(['marker', ...args]); complete = true;},
    }, execFileSync: (...args) => {calls.push(['execute', ...args]); if (failure) throw Error('unpack failed');}};
  vm.runInNewContext(source.slice(source.indexOf('function prepareRuntime('), source.indexOf('function probe(')), context);
  return {calls, prepare: context.prepareRuntime};
}

test('Mac relocates outside app and reuses only completed runtime', () => {
  const {calls, prepare} = fixture();
  const runtime = prepare('/readonly/app/runtime');
  assert.ok(runtime.startsWith(path.join('/user profile', 'bundled-runtime')));
  assert.equal(calls.filter(call => call[0] === 'copy').length, 2);
  assert.ok(calls.filter(call => call[0] === 'copy').every(call => call[3].verbatimSymlinks));
  const execution = calls.find(call => call[0] === 'execute');
  assert.equal(execution[1], path.join(runtime, 'python/bin/python'));
  assert.equal(execution[3].env.PYTHONPATH, undefined);
  assert.equal(execution[3].env.PYTHONHOME, undefined);
  assert.equal(calls.at(-1)[0], 'marker');
  assert.equal(prepare('/readonly/app/runtime'), runtime);
  assert.equal(calls.length, 4);
});

test('failed unpack never marks complete; Windows path unchanged', () => {
  const failed = fixture('darwin', true);
  assert.throws(() => failed.prepare('/source'), /unpack failed/);
  assert.ok(!failed.calls.some(call => call[0] === 'marker'));
  const windows = fixture('win32');
  assert.equal(windows.prepare('C:/runtime'), 'C:/runtime');
  assert.equal(windows.calls.length, 0);
});
