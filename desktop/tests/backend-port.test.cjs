const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const net = require('node:net');

const source = fs.readFileSync(path.join(__dirname, '../main.js'), 'utf8');
const probe = source.match(/^function probeBackendPort\([^\n]*\)[\s\S]+?^}/m)[0];
const select = source.match(/^async function findBackendPort\([^\n]*\)[\s\S]+?^}/m)[0];

test('occupied non-HTTP ports are not considered free', async () => {
  const context = { require };
  vm.runInNewContext(probe, context);
  const server = net.createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    assert.equal(await context.probeBackendPort('127.0.0.1', server.address().port), null);
    assert.ok(await context.probeBackendPort('127.0.0.1', 0) > 0);
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
});

test('reserved candidate range falls back to OS-assigned port', async () => {
  const calls = [];
  const context = { probeBackendPort: async (host, port) => { calls.push([host, port]); return port === 0 ? 49123 : null; } };
  vm.runInNewContext(select, context);
  assert.equal(await context.findBackendPort({ host: '127.0.0.1', port: 7083 }), 49123);
  assert.equal(calls.length, 21);
  assert.deepEqual(calls.at(-1), ['127.0.0.1', 0]);
});

test('first bindable candidate wins and upper port bound is respected', async () => {
  const calls = [];
  const context = { probeBackendPort: async (_host, port) => { calls.push(port); return port === 65535 ? null : 49001; } };
  vm.runInNewContext(select, context);
  assert.equal(await context.findBackendPort({ host: '127.0.0.1', port: 65535 }), 49001);
  assert.deepEqual(calls, [65535, 0]);
});
