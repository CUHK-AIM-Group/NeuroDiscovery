const {app, BrowserWindow, Menu, dialog, ipcMain, shell} = require('electron');
const {spawn, execFileSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const instanceId = require('node:crypto').randomUUID();

app.setName('NeuroDiscovery Human Evaluation');
app.setPath('userData', path.join(app.getPath('appData'), 'NeuroDiscovery-Human-Evaluation'));
const origin = 'http://127.0.0.1:17890';
let backend;
let mainWindow;
let backendError = '';
let stopping = false;
const locked = app.requestSingleInstanceLock();
if (!locked) app.quit();
app.on('second-instance', () => { if (mainWindow) {mainWindow.restore(); mainWindow.focus();} });

function prepareRuntime(source) {
  if (process.platform !== 'darwin') return source;
  const manifest = fs.readFileSync(path.join(source, 'runtime-manifest.json'));
  const fingerprint = require('node:crypto').createHash('sha256').update(manifest).digest('hex');
  const runtime = path.join(app.getPath('userData'), 'bundled-runtime', fingerprint);
  const marker = path.join(runtime, '.conda-unpack-complete');
  if (fs.existsSync(marker)) return runtime;
  fs.mkdirSync(runtime, {recursive: true});
  fs.cpSync(path.join(source, 'python'), path.join(runtime, 'python'), {recursive: true, verbatimSymlinks: true});
  fs.cpSync(path.join(source, 'backend'), path.join(runtime, 'backend'), {recursive: true, verbatimSymlinks: true});
  const env = {...process.env, PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1'};
  delete env.PYTHONPATH;
  delete env.PYTHONHOME;
  execFileSync(path.join(runtime, 'python/bin/python'), [path.join(runtime, 'python/bin/conda-unpack')],
    {cwd: runtime, env, timeout: 180000, stdio: 'pipe'});
  fs.writeFileSync(marker, fingerprint, 'utf8');
  return runtime;
}

function probe() {
  return new Promise(resolve => {
    const request = http.get(`${origin}/api/health`, {timeout: 800}, response => {
      let body = '';
      response.on('data', chunk => {body += chunk;});
      response.on('end', () => {
        try {const health = JSON.parse(body); resolve(health.mode === 'human-evaluation-only' && health.instance_id === instanceId);} catch {resolve(false);}
      });
    });
    request.on('error', () => resolve(false));
    request.on('timeout', () => {request.destroy(); resolve(false);});
  });
}

ipcMain.handle('evaluation:export', async (event, request) => {
  if (!mainWindow || event.sender !== mainWindow.webContents || new URL(event.senderFrame.url).origin !== origin) {
    throw new Error('Invalid export sender');
  }
  const payload = request?.payload;
  if (!payload || typeof payload !== 'object' || !payload.human_evaluation_1 || !payload.human_evaluation_2) {
    throw new Error('Invalid evaluation export');
  }
  const content = JSON.stringify(payload, null, 2);
  if (Buffer.byteLength(content) > 32 * 1024 * 1024) throw new Error('Evaluation export is too large');
  const result = await dialog.showSaveDialog(mainWindow, {title: 'Export human evaluation results',
    defaultPath: path.basename(String(request.defaultFileName || 'NeuroDiscovery-human-evaluation.json')), filters: [{name: 'JSON', extensions: ['json']}]});
  if (result.canceled || !result.filePath) return {canceled: true};
  fs.writeFileSync(result.filePath, `${content}\n`, 'utf8');
  return {canceled: false, path: result.filePath};
});

async function boot() {
  const source = app.isPackaged ? path.join(process.resourcesPath, 'evaluation-runtime') : path.join(__dirname, 'runtime-evaluation');
  const runtime = prepareRuntime(source);
  const python = path.join(runtime, 'python', process.platform === 'win32' ? 'python.exe' : 'bin/python');
  if (!fs.existsSync(python)) throw new Error('Prepare the evaluation-only runtime first.');
  const occupied = await new Promise(resolve => {
    const server = require('node:net').createServer();
    server.once('error', () => resolve(true));
    server.listen(17890, '127.0.0.1', () => server.close(() => resolve(false)));
  });
  if (occupied) throw new Error('Evaluation port 17890 is already in use. Close the other evaluation client before retrying.');
  const env = {...process.env, PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1', NEURODISCOVERY_EVALUATION_INSTANCE: instanceId};
  delete env.PYTHONPATH;
  delete env.PYTHONHOME;
  backend = spawn(python, ['-s', '-m', 'core.web.evaluation_app', '--port', '17890', '--data',
    path.join(app.getPath('userData'), 'evaluation-data')], {
    cwd: path.join(runtime, 'backend'), env, windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'],
  });
  backend.on('error', error => {backendError = error.message;});
  backend.stderr.on('data', chunk => {backendError = (backendError + chunk).slice(-6000);});
  backend.on('exit', () => {
    if (!stopping && mainWindow) {
      dialog.showErrorBox('Evaluation server stopped', 'Saved answers remain on disk. Close and reopen the application.');
      app.quit();
    }
  });
  let ready = false;
  for (let attempt = 0; attempt < 90; attempt += 1) {
    if (backend.exitCode !== null) break;
    if (await probe()) {ready = true; break;}
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  if (!ready) throw new Error(`Unable to start the evaluation server.\n${backendError}`);
  mainWindow = new BrowserWindow({width: 1280, height: 900, minWidth: 800, minHeight: 600,
    webPreferences: {preload: path.join(__dirname, 'evaluation-preload.js'), nodeIntegration: false,
      contextIsolation: true, sandbox: true}});
  mainWindow.webContents.setWindowOpenHandler(({url}) => {
    if (/^https:\/\//i.test(url)) shell.openExternal(url);
    return {action: 'deny'};
  });
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (new URL(url).origin !== origin) event.preventDefault();
  });
  mainWindow.webContents.on('will-frame-navigate', (event) => {
    if (new URL(event.url).origin !== origin) event.preventDefault();
  });
  mainWindow.webContents.session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(process.platform === 'darwin' ? [{role: 'appMenu'}] : []),
    {role: 'editMenu'}, {label: 'View', submenu: [{role: 'zoomIn'}, {role: 'zoomOut'}, {role: 'resetZoom'}, {role: 'togglefullscreen'}]},
  ]));
  await mainWindow.loadURL(origin);
}

if (locked) app.whenReady().then(boot).catch(error => {
  dialog.showErrorBox('Human Evaluation', error.message);
  app.quit();
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', () => {stopping = true; if (backend && !backend.killed) backend.kill();});
