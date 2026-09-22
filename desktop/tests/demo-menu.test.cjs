const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../main.js'), 'utf8');

test('demo menus omit evaluation on Windows and macOS without changing standard menus', () => {
  const menuCode = source.slice(source.indexOf('function expertStudyMenuItem('), source.indexOf("ipcMain.handle('neuroclaw:show-application-menu'"));
  for (const platform of ['win32', 'darwin']) {
    for (const demo of [true, false]) {
      let template;
      const context = {
        DEMO_BUILD: demo, APP_NAME: 'NeuroDiscovery', process: {platform}, mainWindow: null,
        desktopText: english => english, sendMenuAction: () => {},
        settingsMenuItem: () => ({label: 'Settings'}), newChatMenuItem: () => ({label: 'New Chat'}),
        Menu: {buildFromTemplate: value => value, setApplicationMenu: value => { template = value; }},
      };
      vm.createContext(context);
      vm.runInContext(menuCode, context);
      context.setApplicationMenu();
      const labels = template.flatMap(menu => menu.submenu.map(item => item.label));
      assert.equal(labels.includes('Human Evaluation 1'), !demo);
      assert.equal(labels.includes('Human Evaluation 2'), !demo);
      assert.ok(labels.includes('Settings'));
    }
  }
});
