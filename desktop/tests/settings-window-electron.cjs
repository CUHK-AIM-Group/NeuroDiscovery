const { app, BrowserWindow, ipcMain, session, nativeTheme } = require('electron');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');
const staticRoot = path.join(root, 'core/web/static');
const output = path.join(root, 'tmp/settings-window-preview');
app.setPath('userData', fs.mkdtempSync(path.join(os.tmpdir(), 'neurodiscovery-settings-test-')));
app.disableHardwareAcceleration();

const requests = [];
const errors = [];
let desktopConfig = { language: 'English', theme: 'light', llmConnection: 'api', llmProvider: 'openai' };
let configReads = 0;
let configSaves = 0;
let mainWindow;
let settingsWindow;

const server = http.createServer((request, response) => {
  const url = new URL(request.url, 'http://localhost');
  requests.push(`${request.method} ${url.pathname}`);
  const json = data => {
    response.setHeader('Content-Type', 'application/json');
    response.end(JSON.stringify(data));
  };
  if (url.pathname === '/api/workbench/state') return json({ revision: 1, state: { sessions: [], projects: [] } });
  if (url.pathname === '/api/harness') return json({ schema_version: 1, capabilities: [] });
  if (url.pathname === '/api/env') return json({ provider: 'fixture', model: 'fixture', available_models: [] });
  if (url.pathname === '/api/skills') return json({ skills: [] });
  if (url.pathname === '/api/checkpoints') return json({ checkpoints: [] });
  if (url.pathname === '/api/neurooracle/graph/status') return json({ available: false });
  if (url.pathname === '/api/llm/validate') return json({ ok: true });
  if (url.pathname.startsWith('/api/')) return json({ ok: true, items: [] });
  const filename = url.pathname === '/harness' ? 'index.html' : url.pathname.replace(/^\/static\//, '');
  const target = path.resolve(staticRoot, filename);
  if (!target.startsWith(staticRoot + path.sep) || !fs.existsSync(target) || !fs.statSync(target).isFile()) {
    response.statusCode = 404;
    return response.end();
  }
  response.setHeader('Content-Type', ({ '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.png': 'image/png' })[path.extname(target)] || 'application/octet-stream');
  fs.createReadStream(target).pipe(response);
});

function watchErrors(window) {
  window.webContents.on('console-message', details => {
    if (details.level === 'error' && !/ERR_BLOCKED_BY_CLIENT|404/.test(details.message)) errors.push(details.message);
  });
  window.webContents.on('render-process-gone', (_event, details) => errors.push(`Renderer exited: ${details.reason}`));
}

async function until(predicate, message) {
  for (let i = 0; i < 100; i++) {
    if (await predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(message);
}

async function run() {
  await app.whenReady();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  session.defaultSession.webRequest.onBeforeRequest((details, callback) =>
    callback({ cancel: !details.url.startsWith(origin + '/') && !details.url.startsWith('data:') }));

  const desktopSource = fs.readFileSync(path.join(root, 'desktop/main.js'), 'utf8');
  const policy = {
    URL, backendUrl: origin, isBooting: false, process, app, nativeTheme, path,
    __dirname: path.join(root, 'desktop'), APP_NAME: 'NeuroDiscovery fixture',
    APP_ICON_PATH: path.join(root, 'desktop/assets/icon.png'),
    BrowserWindow: class HiddenWindow extends BrowserWindow {
      constructor(options) { super({ ...options, show: false }); }
    },
    focusMainWindow: () => !!mainWindow && !mainWindow.isDestroyed(),
    installContextMenu() {}, boot: () => assert.fail('Restore must reuse the existing backend'),
    log: message => errors.push(message), loadErrorPage: error => { throw error; },
  };
  Object.defineProperties(policy, {
    mainWindow: { get: () => mainWindow, set: value => mainWindow = value },
    settingsWindow: { get: () => settingsWindow, set: value => settingsWindow = value },
  });
  vm.createContext(policy);
  for (const name of ['isTrustedDesktopSender', 'requireTrustedDesktopSender', 'notifyMainSettingsChanged', 'installDesktopNavigation', 'windowChromeOptions', 'createWindow', 'restoreMainWindow']) {
    vm.runInContext(desktopSource.match(new RegExp(`^(?:async )?function ${name}\\([^\\n]*\\)[\\s\\S]+?^}`, 'm'))[0], policy);
  }

  ipcMain.handle('neuroclaw:get-config', event => {
    policy.requireTrustedDesktopSender(event);
    configReads++;
    return { config: desktopConfig, configPath: 'fixture', logsPath: 'fixture', restartRequired: false };
  });
  ipcMain.handle('neuroclaw:subscription-status', () => ({ ok: true, installed: false, signedIn: false }));
  ipcMain.handle('neuroclaw:save-config', (event, config) => {
    policy.requireTrustedDesktopSender(event);
    configSaves++;
    const fields = Object.keys(config).filter(key => JSON.stringify(config[key]) !== JSON.stringify(desktopConfig[key]));
    desktopConfig = config;
    policy.notifyMainSettingsChanged(event, fields);
    return { config: desktopConfig, configPath: 'fixture', restartRequired: false };
  });
  ipcMain.handle('neuroclaw:set-theme', (event, theme) => {
    policy.requireTrustedDesktopSender(event);
    desktopConfig.theme = theme;
    policy.notifyMainSettingsChanged(event, ['theme']);
    return { theme, shouldUseDarkColors: theme === 'dark' };
  });
  ipcMain.handle('neuroclaw:set-language', (event, language) => {
    policy.requireTrustedDesktopSender(event);
    desktopConfig.language = language;
    policy.notifyMainSettingsChanged(event, ['language']);
    return { language };
  });
  ipcMain.handle('neuroclaw:open-settings-window', (_event, target = {}) => {
    if (settingsWindow && !settingsWindow.isDestroyed()) {
      settingsWindow.webContents.send('neuroclaw:settings-navigate', target.section || 'general');
      return true;
    }
    settingsWindow = new BrowserWindow({
      show: false, width: 1160, height: 800,
      webPreferences: { preload: path.join(root, 'desktop/preload.js'), contextIsolation: true, nodeIntegration: false },
    });
    watchErrors(settingsWindow);
    policy.installDesktopNavigation(settingsWindow);
    void settingsWindow.loadURL(`${origin}/harness?settingsWindow=1&section=${encodeURIComponent(target.section || 'general')}`);
    return true;
  });

  mainWindow = new BrowserWindow({
    show: false, width: 1320, height: 900,
    webPreferences: { preload: path.join(root, 'desktop/preload.js'), contextIsolation: true, nodeIntegration: false },
  });
  watchErrors(mainWindow);
  policy.installDesktopNavigation(mainWindow);
  await mainWindow.loadURL(`${origin}/harness`);
  await until(() => mainWindow.webContents.executeJavaScript("document.querySelector('#nav-settings')?.getAttribute('aria-label') === 'Settings'"), 'Main window did not initialize');
  await mainWindow.webContents.executeJavaScript("document.querySelector('#nav-settings').click()");
  await until(() => settingsWindow && !settingsWindow.isDestroyed() && settingsWindow.webContents.executeJavaScript("document.querySelector('#settings-page')?.hidden === false"), 'Separate settings window did not open');

  const layout = await settingsWindow.webContents.executeJavaScript(`(() => ({
    mode: document.documentElement.dataset.settingsWindow,
    navVisible: getComputedStyle(document.querySelector('.settings-sidebar')).display,
    mainSidebarVisible: getComputedStyle(document.querySelector('#app-sidebar')).display,
    headerVisible: getComputedStyle(document.querySelector('#app > header')).display,
    title: document.querySelector('#settings-page .page-title').textContent,
    section: document.querySelector('.settings-nav-item.active')?.dataset.settingsSection,
    usageNav: Boolean(document.querySelector('[data-settings-section="usage"]')),
    usageCommand: Boolean(window.ComposerCommands.commands.find(item => item.command === '/usage')),
    closeButtonVisible: getComputedStyle(document.querySelector('#settings-back-btn')).display !== 'none',
    pageWidth: document.querySelector('#settings-page').getBoundingClientRect().width,
    windowWidth: innerWidth,
  }))()`);
  assert.equal(layout.mode, 'true');
  assert.equal(layout.navVisible, 'flex');
  assert.equal(layout.mainSidebarVisible, 'none');
  assert.equal(layout.headerVisible, 'none');
  assert.equal(layout.section, 'general');
  assert.equal(layout.usageNav, false);
  assert.equal(layout.usageCommand, false);
  assert.equal(layout.closeButtonVisible, false);
  assert.equal(layout.pageWidth, layout.windowWidth);

  const settingsRequestStart = requests.length;
  settingsWindow.webContents.send('neuroclaw:settings-navigate', 'runtime');
  await until(() => settingsWindow.webContents.executeJavaScript("document.querySelector('.settings-nav-item.active')?.dataset.settingsSection === 'runtime'"), 'Settings navigation did not change sections');
  await settingsWindow.webContents.executeJavaScript("document.querySelector('[data-settings-section=general]').click()");
  fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(path.join(output, 'settings-window-light.png'), (await settingsWindow.webContents.capturePage()).toPNG());
  await settingsWindow.webContents.executeJavaScript(`(() => {
    const row = document.querySelector('[data-setting-row="theme"]');
    row.querySelector('[data-settings-select-trigger]').click();
    row.querySelector('[data-settings-select-option="dark"]').click();
  })()`);
  await until(() => settingsWindow.webContents.executeJavaScript("document.documentElement.dataset.theme === 'dark'"), 'Settings window did not apply dark theme');
  await until(() => mainWindow.webContents.executeJavaScript("document.documentElement.dataset.theme === 'dark'"), 'Main window did not receive dark theme');
  fs.writeFileSync(path.join(output, 'settings-window-dark.png'), (await settingsWindow.webContents.capturePage()).toPNG());
  const readsBeforeSave = configReads;
  await settingsWindow.webContents.executeJavaScript("document.querySelector('#settings-save-btn').click()");
  await until(() => configSaves === 1, 'Settings were not saved');
  await until(() => configReads > readsBeforeSave, 'Main window did not refresh configuration after saving');
  assert.ok(!requests.slice(settingsRequestStart).some(item => /POST \/api\/chat/.test(item)), requests.join('\n'));
  assert.deepEqual(errors, []);

  await settingsWindow.webContents.executeJavaScript(`(() => {
    document.querySelector('[data-settings-section=llm]').click();
    document.querySelector('[data-setting-key=llmBaseUrl]').value = 'https://unsaved.example.invalid/v1';
  })()`);
  await mainWindow.webContents.executeJavaScript("applyTheme('light')");
  await until(() => settingsWindow.webContents.executeJavaScript("state.desktopConfig.theme === 'light' && document.documentElement.dataset.theme === 'light'"), 'Reverse theme sync failed');
  assert.equal(await settingsWindow.webContents.executeJavaScript("document.querySelector('[data-setting-key=llmBaseUrl]').value"), 'https://unsaved.example.invalid/v1');
  await settingsWindow.webContents.executeJavaScript("document.querySelector('#settings-save-btn').click()");
  await until(() => configSaves === 2, 'Second settings save failed');
  assert.equal(desktopConfig.theme, 'light');
  await settingsWindow.webContents.executeJavaScript('applyTextScale(1.4, true)');
  await until(() => mainWindow.webContents.executeJavaScript('state.textScale === 1.4'), 'Settings-to-main text size sync failed');
  await mainWindow.webContents.executeJavaScript('applyTextScale(1.3, true)');
  await until(() => settingsWindow.webContents.executeJavaScript('state.textScale === 1.3'), 'Main-to-settings text size sync failed');
  await mainWindow.webContents.executeJavaScript("state.desktopConfig.language = 'Simplified Chinese'; applyLanguage(true)");
  await until(() => settingsWindow.webContents.executeJavaScript("document.documentElement.lang === 'zh-CN'"), 'Reverse language sync failed');

  const markdownChecks = await mainWindow.webContents.executeJavaScript(`(() => {
    const bubble = document.createElement('div'); document.body.appendChild(bubble);
    window.__markdownProbe = 0;
    renderMarkdownBubble(bubble, '[probe](https://example.invalid/" onmouseover="window.__markdownProbe=1)');
    bubble.querySelector('a')?.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
    const fallbackBlocked = window.__markdownProbe === 0 && !bubble.querySelector('[onmouseover]');
    const oldParse = window.marked.parse;
    window.marked.parse = () => '<img src="data:," onerror="window.__markdownProbe=1"><a href="javascript:window.__markdownProbe=1">probe</a><svg onload="window.__markdownProbe=1"></svg><form><input name="neuroclawDesktop"></form><p style="position:fixed">text</p>';
    renderMarkdownBubble(bubble, 'probe');
    const onlineBlocked = !bubble.querySelector('[onerror], [onload], svg, form, input, [style]') && !bubble.querySelector('a').hasAttribute('href');
    window.marked.parse = () => '<h2>Heading</h2><table><tbody><tr><td>Cell</td></tr></tbody></table><pre><code class="language-python">print(1)</code></pre><a href="https://example.invalid">Link</a>';
    renderMarkdownBubble(bubble, 'safe');
    const formattingPreserved = !!bubble.querySelector('h2') && !!bubble.querySelector('.message-table-scroll table') && !!bubble.querySelector('.code-block code') && bubble.querySelector('a').rel === 'noopener noreferrer';
    const oldPurifier = window.DOMPurify; window.DOMPurify = undefined;
    renderMarkdownBubble(bubble, '<img src=x onerror=probe()>');
    const missingSanitizerSafe = !bubble.querySelector('img') && bubble.textContent === '<img src=x onerror=probe()>';
    window.DOMPurify = oldPurifier; window.marked.parse = oldParse; bubble.remove();
    return { fallbackBlocked, onlineBlocked, formattingPreserved, missingSanitizerSafe };
  })()`);
  assert.deepEqual(markdownChecks, { fallbackBlocked: true, onlineBlocked: true, formattingPreserved: true, missingSanitizerSafe: true });

  const oldMainId = mainWindow.id;
  mainWindow.destroy();
  await Promise.all([policy.restoreMainWindow(), policy.restoreMainWindow()]);
  assert.notEqual(mainWindow.id, oldMainId);
  assert.equal(settingsWindow.isDestroyed(), false);
  assert.equal(mainWindow.isVisible(), false);
  await until(() => mainWindow.webContents.executeJavaScript('state.settingsLoaded'), 'Restored main window did not initialize');
  assert.equal(new URL(mainWindow.webContents.getURL()).origin, origin);
  assert.equal(policy.isBooting, false);
  assert.deepEqual(errors, []);

  await settingsWindow.webContents.executeJavaScript('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
  fs.writeFileSync(path.join(output, 'settings-window.png'), (await settingsWindow.webContents.capturePage()).toPNG());
  await settingsWindow.loadURL('data:text/html,<p>Untrusted frame</p>');
  const rejected = await settingsWindow.webContents.executeJavaScript("window.neuroclawDesktop.getConfig().then(() => false, error => error.message.includes('Untrusted desktop IPC sender'))");
  assert.equal(rejected, true);
  fs.writeFileSync(path.join(output, 'verification.json'), JSON.stringify({ layout, markdownChecks, mainWindowRestored: true, untrustedIpcRejected: rejected, configReads, configSaves, requests, errors }, null, 2));
  settingsWindow.destroy();
  mainWindow.destroy();
  console.log(`Settings window verification passed: ${output}`);
}

run().then(() => { server.close(); app.exit(0); }).catch(error => { console.error(error); server.close(); app.exit(1); });
