const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('evaluation native bridge saves the shared export contract and rejects other senders', async () => {
  const source = fs.readFileSync(path.join(__dirname, '../evaluation-main.js'), 'utf8');
  let handler;
  let saved;
  const mainWindow = {webContents: {}};
  const context = {mainWindow, origin: 'http://127.0.0.1:17890', URL, Buffer, path,
    ipcMain: {handle: (name, callback) => {assert.equal(name, 'evaluation:export'); handler = callback;}},
    dialog: {showSaveDialog: async (_window, options) => {
      assert.equal(options.defaultPath, 'participant.json');
      return {canceled: false, filePath: 'test-output.json'};
    }},
    fs: {writeFileSync: (filename, content) => {saved = {filename, content};}},
  };
  vm.runInNewContext(source.slice(source.indexOf("ipcMain.handle('evaluation:export'"), source.indexOf('async function boot()')), context);
  const event = {sender: mainWindow.webContents, senderFrame: {url: context.origin + '/'}};
  const payload = {human_evaluation_1: {sessions: []}, human_evaluation_2: {sessions: []}};
  const result = await handler(event, {defaultFileName: 'participant.json', payload});
  assert.equal(result.canceled, false);
  assert.deepEqual(JSON.parse(saved.content), payload);
  assert.ok(saved.content.endsWith('\n'));
  await assert.rejects(handler({...event, sender: {}}, {payload}), /Invalid export sender/);
  await assert.rejects(handler({...event, senderFrame: {url: 'https://example.com'}}, {payload}), /Invalid export sender/);
  await assert.rejects(handler(event, {payload: {}}), /Invalid evaluation export/);
});
