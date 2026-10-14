const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../main.js'), 'utf8');
const fn = name => source.match(new RegExp(`^(?:async )?function ${name}\\([^\\n]*\\)[\\s\\S]+?^}`, 'm'))[0];

function windowFixture() {
  const main = { isDestroyed: () => false, webContents: { mainFrame: { url: 'http://127.0.0.1:7080/harness' } } };
  const settings = { isDestroyed: () => false, webContents: { mainFrame: { url: 'http://127.0.0.1:7080/harness?settingsWindow=1' } } };
  return { mainWindow: main, settingsWindow: settings, backendUrl: 'http://127.0.0.1:7080', URL };
}

test('desktop IPC allows only its local top frames, before reading or changing configuration', () => {
  const context = windowFixture();
  vm.createContext(context);
  vm.runInContext(fn('isTrustedDesktopSender') + '\n' + fn('requireTrustedDesktopSender'), context);
  const event = owner => ({ sender: owner.webContents, senderFrame: owner.webContents.mainFrame });
  for (const owner of [context.mainWindow, context.settingsWindow]) assert.equal(context.isTrustedDesktopSender(event(owner)), true);
  assert.equal(context.isTrustedDesktopSender(null), false);
  assert.equal(context.isTrustedDesktopSender({ sender: {}, senderFrame: {} }), false);
  const owner = context.mainWindow;
  assert.equal(context.isTrustedDesktopSender({ sender: owner.webContents, senderFrame: { url: owner.webContents.mainFrame.url } }), false);
  for (const url of ['https://example.invalid/harness', 'http://127.0.0.1:7081/harness', 'http://127.0.0.1:7080/api/artifacts/html', 'data:text/html,fixture']) {
    owner.webContents.mainFrame.url = url;
    assert.equal(context.isTrustedDesktopSender(event(owner)), false, url);
  }
  const handlers = {};
  let reads = 0, writes = 0;
  Object.assign(context, { ipcMain: { handle: (key, cb) => handlers[key] = cb }, loadConfig: () => { reads++; }, saveConfig: () => { writes++; } });
  for (const key of ['get-config', 'save-config']) {
    vm.runInContext(source.match(new RegExp(`^ipcMain.handle\\('neuroclaw:${key}'[\\s\\S]+?^\\}\\);`, 'm'))[0], context);
    assert.throws(() => handlers['neuroclaw:' + key](event(owner), {}), /Untrusted/);
  }
  assert.equal(reads, 0);
  assert.equal(writes, 0);
});

test('navigation and redirects cannot leave the desktop workspace or load artifact HTML', () => {
  const listeners = {};
  const context = { URL, backendUrl: 'http://127.0.0.1:7080' };
  vm.runInNewContext(fn('installDesktopNavigation'), context);
  context.installDesktopNavigation({ webContents: { on: (key, cb) => listeners[key] = cb } });
  for (const key of ['will-navigate', 'will-redirect']) {
    for (const url of ['http://127.0.0.1:7080/', 'http://127.0.0.1:7080/harness?settingsWindow=1']) {
      listeners[key]({ preventDefault: () => assert.fail(url) }, url);
    }
    for (const url of ['https://example.invalid', 'data:text/html,x', 'http://127.0.0.1:7080/api/artifacts/html']) {
      let blocked = false;
      listeners[key]({ preventDefault: () => blocked = true }, url);
      assert.equal(blocked, true, url);
    }
  }
  listeners['will-redirect']({ isMainFrame: false, preventDefault: () => assert.fail('Subframe redirect blocked') }, 'http://127.0.0.1:7080/study');
  listeners['will-redirect']({ preventDefault: () => assert.fail('Legacy subframe redirect blocked') }, 'http://127.0.0.1:7080/study', false, false);
});

test('configuration notifications go both ways and skip their sender and destroyed windows', () => {
  const context = windowFixture();
  const calls = [];
  for (const key of ['mainWindow', 'settingsWindow']) context[key].webContents.send = (channel, payload) => calls.push({ key, channel, fields: [...payload.fields] });
  vm.runInNewContext(fn('notifyMainSettingsChanged'), context);
  context.notifyMainSettingsChanged({ sender: context.mainWindow.webContents }, ['theme']);
  context.notifyMainSettingsChanged({ sender: context.settingsWindow.webContents }, ['language']);
  assert.deepEqual(calls.map(call => call.key), ['settingsWindow', 'mainWindow']);
  assert.deepEqual(calls.map(call => call.fields), [['theme'], ['language']]);
  context.settingsWindow.isDestroyed = () => true;
  context.notifyMainSettingsChanged({ sender: context.mainWindow.webContents }, ['theme']);
  assert.equal(calls.length, 2);
});

test('restoring a closed main window reuses the backend even while settings remains open', async () => {
  let focused = false, creates = 0, boots = 0;
  const loaded = [];
  const context = { URL, isBooting: false, backendUrl: 'http://127.0.0.1:7080', process: { argv: [] },
    app: { isPackaged: false }, focusMainWindow: () => focused, boot: () => boots++,
    createWindow: () => { creates++; context.mainWindow = { loadURL: async url => { loaded.push(url); focused = true; } }; },
    log: assert.fail, loadErrorPage: assert.fail };
  vm.createContext(context);
  vm.runInContext(fn('restoreMainWindow'), context);
  await Promise.all([context.restoreMainWindow(), context.restoreMainWindow()]);
  assert.equal(creates, 1);
  assert.equal(boots, 0);
  assert.equal(new URL(loaded[0]).pathname, '/harness');
  assert.equal(context.isBooting, false);
  focused = false;
  context.backendUrl = '';
  await context.restoreMainWindow();
  assert.equal(boots, 1);
  let restores = 0;
  const listeners = {};
  Object.assign(context, { hasSingleInstanceLock: true, restoreMainWindow: () => restores++, stopBackend() {},
    app: { on: (key, cb) => listeners[key] = cb, whenReady: () => ({ then() {} }), quit() {} } });
  vm.runInContext(source.slice(source.lastIndexOf('if (!hasSingleInstanceLock)')), context);
  listeners['second-instance']();
  listeners.activate();
  assert.equal(restores, 2);
});
