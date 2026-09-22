const {app, BrowserWindow} = require('electron');
const {spawn} = require('node:child_process');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const net = require('node:net');

const desktop = path.resolve(__dirname, '..');
const runtime = path.resolve(process.env.DEMO_TEST_RUNTIME || path.join(desktop, 'runtime-demo'));
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'neurodiscovery-demo-smoke-'));
app.setPath('userData', path.join(temporary, 'profile'));
let backend;
let window;
let backendLog = '';
const errors = [];
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function freePort() {
  const server = net.createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

app.whenReady().then(async () => {
  const port = await freePort();
  const origin = `http://127.0.0.1:${port}`;
  const backendRoot = path.join(runtime, 'backend');
  assert.ok(fs.existsSync(path.join(backendRoot, 'DEMO_DISTRIBUTION.json')));
  for (const excluded of ['core/web/static/study.html', 'core/web/static/discovery-study.js', 'core/web/static/evaluation-home.html', 'core/web/evaluation_app.py', 'core/web/study_materials', 'neurooracle/data/user_study', 'neurooracle/src/user_study.py', 'neurooracle/.frozen']) {
    assert.equal(fs.existsSync(path.join(backendRoot, excluded)), false, excluded);
  }
  backend = spawn(path.join(runtime, 'python/python.exe'), ['-B', '-c',
    `import sys; sys.path.insert(0, ${JSON.stringify(backendRoot)}); import uvicorn; from core.web.server import create_app; uvicorn.run(create_app(), host='127.0.0.1', port=${port})`],
    {cwd: temporary, env: {...process.env, PYTHONNOUSERSITE: '1'}, windowsHide: true});
  backend.stdout.on('data', chunk => { backendLog += chunk; });
  backend.stderr.on('data', chunk => { backendLog += chunk; });
  backend.on('error', error => errors.push(String(error)));
  let ready = false;
  for (let attempt = 0; attempt < 120; attempt++) {
    try { ready = (await fetch(`${origin}/api/distribution/demo`)).status === 200; } catch {}
    if (ready || backend.exitCode !== null) break;
    await delay(500);
  }
  assert.ok(ready, backendLog);
  for (const route of ['/study', '/discovery-study', '/api/studies/config', '/static/study.html']) {
    assert.equal((await fetch(origin + route)).status, 404, route);
  }
  window = new BrowserWindow({show: false, width: 1440, height: 1000, webPreferences: {contextIsolation: true, nodeIntegration: false}});
  window.webContents.on('console-message', (_event, level, message) => {
    if (level >= 3) errors.push(message);
  });
  await window.loadURL(origin);
  await delay(3000);
  assert.equal(await window.webContents.executeJavaScript('HUMAN_EVALUATION_ENABLED'), false);
  await window.webContents.executeJavaScript("setActiveView('expert-study'); setActiveView('hypothesis-ranking'); setActiveView('study-results');");
  assert.equal(await window.webContents.executeJavaScript('state.activeView'), 'chat');
  assert.equal(await window.webContents.executeJavaScript("document.querySelector('#case-study-welcome').hidden"), true);
  await window.webContents.executeJavaScript("setActiveView('settings')");
  assert.equal(await window.webContents.executeJavaScript('state.activeView'), 'settings');
  await window.webContents.executeJavaScript("setActiveView('chat')");
  const screenshot = await window.webContents.capturePage();
  fs.writeFileSync(path.join(temporary, 'demo.png'), screenshot.toPNG());
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ok: true, runtime, screenshot: path.join(temporary, 'demo.png'), checks: ['packaged exclusions', 'backend startup', 'evaluation 404', 'evaluation views disabled', 'settings', 'no renderer errors']}));
}).catch(error => {
  console.error(error, backendLog);
  process.exitCode = 1;
}).finally(() => {
  if (window) window.destroy();
  if (backend) backend.kill();
  app.exit(process.exitCode || 0);
});
