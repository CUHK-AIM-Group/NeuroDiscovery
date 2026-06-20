const { app, BrowserWindow, Menu, dialog, ipcMain, shell, clipboard } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');

const APP_NAME = 'NeuroClaw';
const STARTUP_TIMEOUT_MS = 90_000;
const BUNDLED_RUNTIME_VERSION = '0.2.0';

let mainWindow = null;
let backendProcess = null;
let backendStartedByDesktop = false;
let backendUrl = '';
let logStream = null;
let isBooting = false;

const hasSingleInstanceLock = app.requestSingleInstanceLock();

function focusMainWindow() {
  if (!mainWindow || mainWindow.isDestroyed()) return false;
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.setSkipTaskbar(false);
  mainWindow.show();
  mainWindow.focus();
  if (process.platform === 'win32') {
    const wasAlwaysOnTop = mainWindow.isAlwaysOnTop();
    mainWindow.setAlwaysOnTop(true, 'floating');
    mainWindow.show();
    mainWindow.focus();
    mainWindow.setAlwaysOnTop(wasAlwaysOnTop);
  }
  return true;
}

function isChineseDesktopUi() {
  const raw = String(loadConfig().language || '').toLowerCase();
  if (raw.includes('chinese') || raw.includes('zh') || raw.includes('中文') || raw.includes('简体')) {
    return true;
  }
  if (raw.includes('english') || raw.includes('en')) {
    return false;
  }
  return String(app.getLocale() || '').toLowerCase().startsWith('zh');
}

function desktopText(en, zh) {
  return isChineseDesktopUi() ? zh : en;
}

function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function desktopDataUrl(html) {
  return `data:text/html;charset=utf-8,${encodeURIComponent(html)}`;
}

function startupPageHtml(status, detail = '') {
  const title = escapeHtml(APP_NAME);
  const message = escapeHtml(status || desktopText('Starting NeuroClaw', '正在启动 NeuroClaw'));
  const subtext = escapeHtml(detail || desktopText('Checking the local backend and runtime...', '正在检查本地后端和运行环境...'));
  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${title}</title>
  <style>
    :root {
      color-scheme: light;
      font-family: "Segoe UI", "Microsoft YaHei UI", Arial, sans-serif;
      background: #eef4f6;
      color: #162638;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(760px 360px at 50% 42%, rgba(91, 184, 198, .16), transparent 70%),
        linear-gradient(180deg, #f8fcfc, #eef4f6);
    }
    .startup {
      width: min(520px, calc(100vw - 64px));
      text-align: center;
      display: grid;
      gap: 14px;
      justify-items: center;
    }
    .mark {
      width: 58px;
      height: 58px;
      border-radius: 18px;
      display: grid;
      place-items: center;
      background: #0f7f91;
      color: white;
      font-size: 28px;
      font-weight: 800;
      box-shadow: 0 18px 42px rgba(15, 127, 145, .22);
    }
    .spinner {
      width: 26px;
      height: 26px;
      border-radius: 999px;
      border: 3px solid rgba(15, 127, 145, .18);
      border-top-color: #0f7f91;
      animation: spin .8s linear infinite;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    p {
      margin: 0;
      color: #637484;
      font-size: 14px;
      line-height: 1.7;
    }
    .status {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 28px;
      padding: 0 13px;
      border: 1px solid #c9dde2;
      border-radius: 999px;
      background: rgba(255, 255, 255, .72);
      color: #0a6370;
      font-size: 12px;
      font-weight: 700;
      line-height: 1;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <main class="startup">
    <div class="mark">N</div>
    <div class="spinner" aria-hidden="true"></div>
    <h1>${title}</h1>
    <div class="status">${message}</div>
    <p>${subtext}</p>
  </main>
</body>
</html>`;
}

async function loadStartupPage(status, detail) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  await mainWindow.loadURL(desktopDataUrl(startupPageHtml(status, detail)));
}

async function loadErrorPage(err) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const message = escapeHtml(String(err && (err.stack || err.message) ? (err.stack || err.message) : err));
  await mainWindow.loadURL(desktopDataUrl(`<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(APP_NAME)} failed to start</title>
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      padding: 48px;
      box-sizing: border-box;
      font-family: "Segoe UI", "Microsoft YaHei UI", Arial, sans-serif;
      background: #eef4f6;
      color: #162638;
    }
    main {
      max-width: 880px;
      margin: 0 auto;
      border: 1px solid #d7e4e7;
      border-radius: 16px;
      background: white;
      padding: 24px;
      box-shadow: 0 16px 40px rgba(25, 42, 62, .08);
    }
    h1 { margin: 0 0 12px; font-size: 24px; }
    p { color: #637484; }
    pre {
      overflow: auto;
      white-space: pre-wrap;
      padding: 14px;
      border-radius: 12px;
      background: #f3f7f8;
      border: 1px solid #d7e4e7;
    }
  </style>
</head>
<body>
  <main>
    <h1>${escapeHtml(desktopText('NeuroClaw failed to start', 'NeuroClaw 启动失败'))}</h1>
    <p>${escapeHtml(path.join(app.getPath('userData'), 'logs'))}</p>
    <pre>${message}</pre>
  </main>
</body>
</html>`));
}

function repoRoot() {
  return path.resolve(__dirname, '..');
}

function userConfigPath() {
  return path.join(app.getPath('userData'), 'desktop-config.json');
}

function packagedRuntimeSourceRoot() {
  return path.join(process.resourcesPath || __dirname, 'runtime');
}

function userRuntimeRoot() {
  return path.join(app.getPath('userData'), 'bundled-runtime', BUNDLED_RUNTIME_VERSION);
}

function bundledPythonExe(runtimeRoot = userRuntimeRoot()) {
  if (process.platform === 'win32') return path.join(runtimeRoot, 'python', 'python.exe');
  return path.join(runtimeRoot, 'python', 'bin', 'python');
}

function bundledCondaUnpackExe(runtimeRoot = userRuntimeRoot()) {
  if (process.platform === 'win32') return path.join(runtimeRoot, 'python', 'Scripts', 'conda-unpack.exe');
  return path.join(runtimeRoot, 'python', 'bin', 'conda-unpack');
}

function bundledBackendRoot(runtimeRoot = userRuntimeRoot()) {
  return path.join(runtimeRoot, 'backend');
}

function bundledRuntimeSourceExists() {
  const sourceRoot = packagedRuntimeSourceRoot();
  return fs.existsSync(path.join(sourceRoot, 'python'))
    && fs.existsSync(path.join(sourceRoot, 'backend', 'core', 'agent', 'main.py'));
}

function bundledRuntimeReady(runtimeRoot = userRuntimeRoot()) {
  return fs.existsSync(bundledPythonExe(runtimeRoot))
    && fs.existsSync(path.join(bundledBackendRoot(runtimeRoot), 'core', 'agent', 'main.py'));
}

function copyDirectoryFresh(source, target) {
  fs.rmSync(target, { recursive: true, force: true });
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.cpSync(source, target, { recursive: true });
}

function runBundledCondaUnpack(runtimeRoot) {
  const unpackExe = bundledCondaUnpackExe(runtimeRoot);
  const marker = path.join(runtimeRoot, '.conda-unpack-complete');
  if (!fs.existsSync(unpackExe) || fs.existsSync(marker)) return;
  const proc = require('node:child_process').spawnSync(unpackExe, [], {
    cwd: path.join(runtimeRoot, 'python'),
    windowsHide: true,
    encoding: 'utf8',
  });
  if (proc.status !== 0) {
    throw new Error(`Bundled Python post-install failed: ${proc.stderr || proc.stdout || `exit ${proc.status}`}`);
  }
  fs.writeFileSync(marker, new Date().toISOString(), 'utf8');
}

function ensureBundledRuntime() {
  if (!app.isPackaged || !bundledRuntimeSourceExists()) return null;
  const sourceRoot = packagedRuntimeSourceRoot();
  const runtimeRoot = userRuntimeRoot();
  const marker = path.join(runtimeRoot, '.runtime-version');
  const currentVersion = fs.existsSync(marker) ? fs.readFileSync(marker, 'utf8').trim() : '';
  if (!bundledRuntimeReady(runtimeRoot) || currentVersion !== BUNDLED_RUNTIME_VERSION) {
    log(`Preparing bundled runtime ${BUNDLED_RUNTIME_VERSION} at ${runtimeRoot}`);
    fs.rmSync(runtimeRoot, { recursive: true, force: true });
    fs.mkdirSync(runtimeRoot, { recursive: true });
    copyDirectoryFresh(path.join(sourceRoot, 'python'), path.join(runtimeRoot, 'python'));
    copyDirectoryFresh(path.join(sourceRoot, 'backend'), path.join(runtimeRoot, 'backend'));
    fs.writeFileSync(marker, BUNDLED_RUNTIME_VERSION, 'utf8');
  }
  runBundledCondaUnpack(runtimeRoot);
  return {
    pythonExe: bundledPythonExe(runtimeRoot),
    repoRoot: bundledBackendRoot(runtimeRoot),
  };
}

function firstExistingPath(paths) {
  return paths.find(candidate => candidate && fs.existsSync(candidate)) || paths[0] || '';
}

function defaultCondaExe(home) {
  if (process.platform === 'win32') {
    return path.join(home, 'anaconda3', 'Scripts', 'conda.exe');
  }
  return firstExistingPath([
    path.join(home, 'miniforge3', 'bin', 'conda'),
    path.join(home, 'miniconda3', 'bin', 'conda'),
    path.join(home, 'anaconda3', 'bin', 'conda'),
  ]);
}

function defaultPythonExe(home) {
  if (process.platform === 'win32') {
    return path.join(home, 'anaconda3', 'envs', 'neuroclaw', 'python.exe');
  }
  return firstExistingPath([
    path.join(home, 'miniforge3', 'envs', 'neuroclaw', 'bin', 'python'),
    path.join(home, 'miniconda3', 'envs', 'neuroclaw', 'bin', 'python'),
    path.join(home, 'anaconda3', 'envs', 'neuroclaw', 'bin', 'python'),
  ]);
}

function defaultConfig() {
  const home = os.homedir();
  const hasBundledRuntime = app.isPackaged && bundledRuntimeSourceExists();
  return {
    host: '127.0.0.1',
    port: 7080,
    runtimeMode: process.env.NEUROCLAW_RUNTIME_MODE || (hasBundledRuntime ? 'bundled' : 'conda'),
    pythonExe: process.env.NEUROCLAW_PYTHON_EXE || defaultPythonExe(home),
    condaExe: process.env.NEUROCLAW_CONDA_EXE || defaultCondaExe(home),
    condaEnv: process.env.NEUROCLAW_CONDA_ENV || 'neuroclaw',
    fslDir: process.env.FSLDIR || '',
    language: process.env.NEUROCLAW_LANGUAGE || 'English',
    proxyUrl: process.env.NEUROCLAW_PROXY_URL || process.env.HTTPS_PROXY || process.env.HTTP_PROXY || 'http://127.0.0.1:7897',
    repoRoot: process.env.NEUROCLAW_REPO_ROOT || (hasBundledRuntime ? bundledBackendRoot() : (app.isPackaged ? path.join(home, 'Documents', 'Code', 'NeuroClaw') : repoRoot())),
  };
}

function loadConfig() {
  const defaults = defaultConfig();
  try {
    const raw = fs.readFileSync(userConfigPath(), 'utf8');
    return { ...defaults, ...JSON.parse(raw) };
  } catch (_err) {
    return defaults;
  }
}

function saveConfig(nextConfig) {
  const defaults = defaultConfig();
  const current = loadConfig();
  const allowed = ['host', 'port', 'runtimeMode', 'pythonExe', 'condaExe', 'condaEnv', 'repoRoot', 'fslDir', 'language', 'proxyUrl'];
  const clean = { ...current };
  for (const key of allowed) {
    if (Object.prototype.hasOwnProperty.call(nextConfig || {}, key)) {
      clean[key] = key === 'port' ? Number(nextConfig[key]) || defaults.port : String(nextConfig[key] || '').trim();
    }
  }
  fs.mkdirSync(path.dirname(userConfigPath()), { recursive: true });
  fs.writeFileSync(userConfigPath(), JSON.stringify({ ...defaults, ...clean }, null, 2), 'utf8');
  return loadConfig();
}

function ensureLogStream() {
  if (logStream) return logStream;
  const logDir = path.join(app.getPath('userData'), 'logs');
  fs.mkdirSync(logDir, { recursive: true });
  logStream = fs.createWriteStream(path.join(logDir, 'neuroclaw-desktop.log'), { flags: 'a' });
  log(`=== ${APP_NAME} desktop start ${new Date().toISOString()} ===`);
  return logStream;
}

function log(message) {
  const line = `[${new Date().toISOString()}] ${message}\n`;
  ensureLogStream().write(line);
}

function requestHealth(url, timeoutMs = 1500) {
  return new Promise((resolve) => {
    const req = http.get(`${url}/api/health`, { timeout: timeoutMs }, (res) => {
      res.resume();
      resolve(res.statusCode >= 200 && res.statusCode < 500);
    });
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
    req.on('error', () => resolve(false));
  });
}

function requestStatusCode(url, pathname, timeoutMs = 1500) {
  return new Promise((resolve) => {
    const req = http.get(`${url}${pathname}`, { timeout: timeoutMs }, (res) => {
      res.resume();
      resolve(res.statusCode || 0);
    });
    req.on('timeout', () => {
      req.destroy();
      resolve(0);
    });
    req.on('error', () => resolve(0));
  });
}

async function requestDesktopCompatible(url) {
  if (!(await requestHealth(url))) return false;
  const graphStatus = await requestStatusCode(url, '/api/neurooracle/graph/status');
  return graphStatus >= 200 && graphStatus < 500 && graphStatus !== 404;
}

async function findBackendPort(config) {
  const base = Number(config.port) || 7080;
  for (let offset = 0; offset < 20; offset += 1) {
    const port = base + offset;
    const url = `http://${config.host}:${port}`;
    if (!(await requestHealth(url))) return port;
  }
  throw new Error(`No free local backend port found from ${base} to ${base + 19}`);
}

async function waitForBackend(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await requestHealth(url)) return true;
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
  return false;
}

function validateConfig(config) {
  if (config.runtimeMode === 'bundled') {
    if (!fs.existsSync(config.repoRoot)) {
      throw new Error(`Bundled NeuroClaw backend not found: ${config.repoRoot}`);
    }
    if (!fs.existsSync(config.pythonExe)) {
      throw new Error(`Bundled Python executable not found: ${config.pythonExe}`);
    }
    return;
  }
  if (config.runtimeMode === 'python') {
    if (!fs.existsSync(config.repoRoot)) {
      throw new Error(`NeuroClaw repo root not found: ${config.repoRoot}`);
    }
    if (!fs.existsSync(config.pythonExe)) {
      throw new Error(`Python executable not found: ${config.pythonExe}`);
    }
    return;
  }
  if (!fs.existsSync(config.repoRoot)) {
    throw new Error(`NeuroClaw repo root not found: ${config.repoRoot}`);
  }
  if (!fs.existsSync(config.condaExe)) {
    throw new Error(`Conda executable not found: ${config.condaExe}`);
  }
}

function resolveRuntimeConfig(config) {
  if (config.runtimeMode !== 'bundled') return config;
  const runtime = ensureBundledRuntime();
  if (!runtime) {
    throw new Error('Bundled runtime is not available in this build.');
  }
  return {
    ...config,
    pythonExe: runtime.pythonExe,
    repoRoot: runtime.repoRoot,
  };
}

async function ensureBackend() {
  const config = resolveRuntimeConfig(loadConfig());
  validateConfig(config);
  backendUrl = `http://${config.host}:${config.port}`;

  if (await requestDesktopCompatible(backendUrl)) {
    log(`Reusing existing NeuroClaw backend at ${backendUrl}`);
    backendStartedByDesktop = false;
    return { url: backendUrl, reused: true };
  }
  if (await requestHealth(backendUrl)) {
    log(`Existing backend at ${backendUrl} is missing desktop APIs; starting a compatible backend on another port`);
  }

  const selectedPort = await findBackendPort(config);
  backendUrl = `http://${config.host}:${selectedPort}`;

  const backendArgs = [
    path.join('core', 'agent', 'main.py'),
    '--web',
    '--port',
    String(selectedPort),
    '--host',
    config.host,
  ];
  const env = { ...process.env };
  if (config.fslDir) env.FSLDIR = config.fslDir;
  if (config.language && config.language !== 'System default') env.NEUROCLAW_LANGUAGE = config.language;
  if (config.proxyUrl) {
    env.NEUROCLAW_PROXY_URL = config.proxyUrl;
    env.HTTP_PROXY = config.proxyUrl;
    env.HTTPS_PROXY = config.proxyUrl;
    env.ALL_PROXY = config.proxyUrl;
    env.NO_PROXY = env.NO_PROXY || '127.0.0.1,localhost';
  }

  const command = config.runtimeMode === 'python' || config.runtimeMode === 'bundled' ? config.pythonExe : config.condaExe;
  const args = config.runtimeMode === 'python' || config.runtimeMode === 'bundled'
    ? backendArgs
    : ['run', '-n', config.condaEnv, 'python', ...backendArgs];

  backendProcess = spawn(command, args, {
    cwd: config.repoRoot,
    env,
    windowsHide: true,
  });
  backendStartedByDesktop = true;
  log(`Started backend pid=${backendProcess.pid} command="${command} ${args.join(' ')}" cwd="${config.repoRoot}"`);

  backendProcess.stdout.on('data', (chunk) => log(`[backend stdout] ${chunk.toString().trimEnd()}`));
  backendProcess.stderr.on('data', (chunk) => log(`[backend stderr] ${chunk.toString().trimEnd()}`));
  backendProcess.on('exit', (code, signal) => {
    log(`Backend exited code=${code} signal=${signal || ''}`);
    backendProcess = null;
  });

  const ready = await waitForBackend(backendUrl, STARTUP_TIMEOUT_MS);
  if (!ready) {
    throw new Error(`NeuroClaw backend did not become ready at ${backendUrl} within ${STARTUP_TIMEOUT_MS / 1000}s`);
  }
  return { url: backendUrl, reused: false };
}

function createWindow() {
  if (focusMainWindow()) return mainWindow;
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 900,
    minWidth: 960,
    minHeight: 680,
    title: APP_NAME,
    backgroundColor: '#eef4f6',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
  mainWindow.once('ready-to-show', () => {
    focusMainWindow();
  });

  installContextMenu(mainWindow);
  return mainWindow;
}

function compactMenuTemplate(template) {
  const items = [];
  for (const item of template) {
    if (!item) continue;
    if (item.type === 'separator') {
      if (items.length && items[items.length - 1].type !== 'separator') {
        items.push(item);
      }
      continue;
    }
    items.push(item);
  }
  while (items.length && items[items.length - 1].type === 'separator') items.pop();
  return items;
}

function installContextMenu(window) {
  window.webContents.on('context-menu', (_event, params) => {
    const editFlags = params.editFlags || {};
    const selectedText = String(params.selectionText || '').trim();
    const hasSelection = selectedText.length > 0;
    const hasLink = Boolean(params.linkURL);
    const hasImage = params.mediaType === 'image' || Boolean(params.srcURL);

    const template = [];

    if (params.isEditable) {
      template.push(
        { label: desktopText('Undo', '撤销'), role: 'undo', enabled: Boolean(editFlags.canUndo) },
        { label: desktopText('Redo', '重做'), role: 'redo', enabled: Boolean(editFlags.canRedo) },
        { type: 'separator' },
        { label: desktopText('Cut', '剪切'), role: 'cut', enabled: Boolean(editFlags.canCut) },
        { label: desktopText('Copy', '复制'), role: 'copy', enabled: Boolean(editFlags.canCopy || hasSelection) },
        { label: desktopText('Paste', '粘贴'), role: 'paste', enabled: Boolean(editFlags.canPaste) },
        { label: desktopText('Delete', '删除'), role: 'delete', enabled: Boolean(editFlags.canDelete) },
        { type: 'separator' },
        { label: desktopText('Select All', '全选'), role: 'selectAll', enabled: Boolean(editFlags.canSelectAll) },
      );
    } else {
      if (hasSelection) {
        template.push(
          { label: desktopText('Copy', '复制'), role: 'copy' },
          { type: 'separator' },
        );
      }

      if (hasLink) {
        template.push(
          {
            label: desktopText('Open Link', '打开链接'),
            click: () => shell.openExternal(params.linkURL),
          },
          {
            label: desktopText('Copy Link', '复制链接'),
            click: () => clipboard.writeText(params.linkURL),
          },
          { type: 'separator' },
        );
      }

      if (hasImage) {
        template.push(
          {
            label: desktopText('Copy Image', '复制图片'),
            click: () => window.webContents.copyImageAt(params.x, params.y),
          },
        );
        if (params.srcURL) {
          template.push({
            label: desktopText('Copy Image Address', '复制图片地址'),
            click: () => clipboard.writeText(params.srcURL),
          });
        }
        template.push({ type: 'separator' });
      }

      template.push(
        { label: desktopText('Select All', '全选'), role: 'selectAll' },
        { type: 'separator' },
        {
          label: desktopText('New Chat', '新建对话'),
          click: () => sendMenuAction('new-chat'),
        },
        {
          label: desktopText('Settings...', '设置...'),
          click: () => sendMenuAction('open-settings'),
        },
        { type: 'separator' },
        { label: desktopText('Reload', '重新加载'), role: 'reload' },
      );
    }

    if (!app.isPackaged) {
      template.push(
        { type: 'separator' },
        {
          label: desktopText('Inspect Element', '检查元素'),
          click: () => window.webContents.inspectElement(params.x, params.y),
        },
      );
    }

    const menu = Menu.buildFromTemplate(compactMenuTemplate(template));
    menu.popup({ window });
  });
}

function sendMenuAction(action) {
  const target = BrowserWindow.getFocusedWindow() || mainWindow;
  if (target && !target.isDestroyed()) {
    target.webContents.send('neuroclaw:menu-action', action);
  }
}

function setApplicationMenu() {
  const template = [
    {
      label: APP_NAME,
      submenu: [
        {
          label: desktopText('New Chat', '新建对话'),
          accelerator: 'CmdOrCtrl+N',
          click: () => sendMenuAction('new-chat'),
        },
        { type: 'separator' },
        {
          label: desktopText('Settings...', '设置...'),
          click: () => sendMenuAction('open-settings'),
        },
        { type: 'separator' },
        { label: desktopText('Reload', '重新加载'), role: 'reload', accelerator: 'CmdOrCtrl+R' },
        { type: 'separator' },
        { label: desktopText('Exit', '退出'), role: 'quit' },
      ],
    },
    {
      label: desktopText('Edit', '编辑'),
      submenu: [
        { label: desktopText('Undo', '撤销'), role: 'undo' },
        { label: desktopText('Redo', '重做'), role: 'redo' },
        { type: 'separator' },
        { label: desktopText('Cut', '剪切'), role: 'cut' },
        { label: desktopText('Copy', '复制'), role: 'copy' },
        { label: desktopText('Paste', '粘贴'), role: 'paste' },
        { label: desktopText('Select All', '全选'), role: 'selectAll' },
      ],
    },
    {
      label: desktopText('View', '视图'),
      submenu: [
        { label: desktopText('Reload', '重新加载'), role: 'reload' },
        { label: desktopText('Force Reload', '强制重新加载'), role: 'forceReload' },
        { type: 'separator' },
        { label: desktopText('Actual Size', '实际大小'), role: 'resetZoom' },
        { label: desktopText('Zoom In', '放大'), role: 'zoomIn' },
        { label: desktopText('Zoom Out', '缩小'), role: 'zoomOut' },
        { type: 'separator' },
        { label: desktopText('Toggle Full Screen', '切换全屏'), role: 'togglefullscreen' },
      ],
    },
    {
      label: desktopText('Help', '帮助'),
      submenu: [
        {
          label: desktopText('About NeuroClaw', '关于 NeuroClaw'),
          click: () => dialog.showMessageBox({
            type: 'info',
            title: desktopText(`About ${APP_NAME}`, `关于 ${APP_NAME}`),
            message: `${APP_NAME} Desktop`,
            detail: desktopText(
              `Version ${app.getVersion()}\nBackend: ${backendUrl || 'not started'}`,
              `版本 ${app.getVersion()}\n后端：${backendUrl || '未启动'}`,
            ),
          }),
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

ipcMain.handle('neuroclaw:get-config', () => ({
  config: loadConfig(),
  configPath: userConfigPath(),
  logsPath: path.join(app.getPath('userData'), 'logs'),
}));

ipcMain.handle('neuroclaw:save-config', (_event, config) => ({
  config: saveConfig(config),
  configPath: userConfigPath(),
  restartRequired: true,
}));

ipcMain.handle('neuroclaw:restart', () => {
  log('Restart requested from settings');
  stopBackend();
  app.relaunch();
  app.exit(0);
  return { ok: true };
});

async function boot() {
  if (focusMainWindow() || isBooting) return;
  isBooting = true;
  try {
    createWindow();
    setApplicationMenu();
    await loadStartupPage(
      desktopText('Starting local runtime', '正在启动本地运行环境'),
      desktopText('Checking backend health, Python, and configured paths.', '正在检查后端健康状态、Python 和已配置路径。'),
    );
    const backend = await ensureBackend();
    log(`Loading ${backend.url} reused=${backend.reused}`);
    await loadStartupPage(
      desktopText('Loading workspace', '正在加载工作台'),
      backend.reused
        ? desktopText('Connected to an existing NeuroClaw backend.', '已连接到正在运行的 NeuroClaw 后端。')
        : desktopText('The backend is ready. Opening the desktop UI.', '后端已就绪，正在打开桌面界面。'),
    );
    await mainWindow.loadURL(backend.url);
    focusMainWindow();
  } catch (err) {
    log(`Startup failed: ${err.stack || err.message || err}`);
    dialog.showErrorBox(
      'NeuroClaw failed to start',
      `${err.message || err}\n\nLogs: ${path.join(app.getPath('userData'), 'logs')}`,
    );
    await loadErrorPage(err);
  } finally {
    isBooting = false;
  }
}

function stopBackend() {
  if (backendStartedByDesktop && backendProcess && !backendProcess.killed) {
    log(`Stopping backend pid=${backendProcess.pid}`);
    backendProcess.kill();
  }
  if (logStream) {
    log(`=== ${APP_NAME} desktop stop ${new Date().toISOString()} ===`);
    logStream.end();
    logStream = null;
  }
}

if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    focusMainWindow();
  });

  app.whenReady().then(boot);

  app.on('activate', () => {
    if (!focusMainWindow() && BrowserWindow.getAllWindows().length === 0) boot();
  });

  app.on('before-quit', stopBackend);

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
  });
}
