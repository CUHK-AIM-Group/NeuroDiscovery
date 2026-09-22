const {app, BrowserWindow} = require('electron');
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');

const root = path.resolve(__dirname, '..');
const runtime = process.env.EVALUATION_TEST_RUNTIME || path.join(root, 'runtime-evaluation');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'evaluation-ui-'));
app.setPath('userData', path.join(temporary, 'profile'));
const origin = 'http://127.0.0.1:17891';
let backend;
let window;
const errors = [];
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function evaluate(source) {
  try {
    return await window.webContents.executeJavaScript(source.startsWith('const ') ? `(()=>{${source}})()` : source, true);
  } catch (error) {
    throw new Error(`${source}: ${error.message}; ${errors.join('; ')}`);
  }
}

async function until(source) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (await evaluate(source)) return;
    await delay(100);
  }
  throw new Error(`UI condition timed out: ${source}`);
}

app.whenReady().then(async () => {
  const env = {...process.env, PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1'};
  delete env.PYTHONPATH;
  delete env.PYTHONHOME;
  backend = spawn(path.join(runtime, 'python/python.exe'), [
    '-s', '-m', 'core.web.evaluation_app', '--port', '17891', '--data', path.join(temporary, 'answers'),
  ], {cwd: path.join(runtime, 'backend'), env, windowsHide: true, stdio: ['ignore', 'ignore', 'pipe']});
  let diagnostics = '';
  backend.stderr.on('data', chunk => {diagnostics += chunk;});
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (backend.exitCode !== null) throw new Error(diagnostics);
    try {if ((await fetch(`${origin}/api/health`)).ok) break;} catch {}
    await delay(100);
  }
  window = new BrowserWindow({show: false, webPreferences: {sandbox: true, contextIsolation: true}});
  window.webContents.on('console-message', (_event, level, message) => {
    if (level >= 3) errors.push(message);
  });
  await window.loadURL(origin);
  assert.equal(await evaluate("document.querySelectorAll('[data-study]').length"), 2);
  await evaluate("document.querySelector('[data-study=discovery-study]').click()");
  await until("Boolean(document.querySelector('iframe').contentDocument?.querySelector('#start'))");
  assert.ok(await evaluate("document.querySelector('iframe').contentDocument.body.textContent.includes('Human Evaluation')"));
  await evaluate("window.postMessage({type:'neurodiscovery:close-study-workspace'},location.origin)");
  assert.equal(await evaluate("document.querySelector('#home').hidden"), true);
  await evaluate("const doc=document.querySelector('iframe').contentDocument;doc.querySelector('[data-discovery-language=en]').click()");
  await until("document.querySelector('iframe').contentDocument.documentElement.lang === 'en'");
  await evaluate("const doc=document.querySelector('iframe').contentDocument;doc.querySelector('[data-discovery-language=zh]').click()");
  await until("document.querySelector('iframe').contentDocument.querySelector('#assignment').options.length > 1");
  await evaluate("const doc=document.querySelector('iframe').contentDocument;doc.querySelector('[name=code]').value='TEST-UI';doc.querySelector('[name=experience]').value='3-5';doc.querySelector('#assignment').value='P01';doc.querySelector('#setup-form').requestSubmit()");
  await until("document.querySelector('iframe').contentDocument.querySelector('#workspace').hidden === false");
  await evaluate("document.querySelector('iframe').contentDocument.querySelector('#pause').click()");
  await until("document.querySelector('#home').hidden === false");
  await evaluate("document.querySelector('[data-study=discovery-study]').click()");
  await until("document.querySelector('iframe').contentDocument?.querySelector('#resume-banner')?.hidden === false");
  await window.loadURL(`${origin}/study?embedded=1`);
  await until("document.querySelector('#assignment').options.length > 1");
  assert.equal(await evaluate("document.querySelector('[data-view=results]').hidden"), true);
  assert.equal(await evaluate("document.querySelector('.setup-advanced').hidden"), true);
  await evaluate("document.querySelector('[data-study-language=zh]').click()");
  await until("document.documentElement.lang.startsWith('zh')");
  await evaluate("document.querySelector('[data-study-language=en]').click()");
  await until("document.documentElement.lang === 'en'");
  await evaluate("document.querySelector('#participant-id').value='TEST-UI';document.querySelector('#participant-experience').value='3-5';document.querySelector('#assignment').value='P01';document.querySelector('#setup-form').requestSubmit()");
  await until("document.querySelector('#workbench-view').classList.contains('active')");
  assert.ok(await evaluate("document.querySelector('#workbench-view').textContent.length > 100"));
  assert.deepEqual(errors, []);
  await evaluate("document.querySelector('#save-progress').click()");
  await delay(500);
  await window.loadURL(`${origin}/study?embedded=1`);
  await until("document.querySelector('#resume-session-btn').offsetParent !== null");
  console.log(JSON.stringify({status: 'passed', home: true, bilingual: true, he1_save_exit_resume: true, he2_save_resume: true, organizer_hidden: true, test_data: temporary}));
}).catch(error => {console.error(error); process.exitCode = 1;}).finally(() => {
  if (backend) backend.kill();
  if (window) window.destroy();
  app.exit(process.exitCode || 0);
});
