const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'evaluation-browser-test-'));
const packageRoot = path.resolve(process.argv[2] || path.join(__dirname, '../dist-evaluation-browser/NeuroDiscovery-Human-Evaluation-1.0.0-Browser-win-x64'));
const edge = process.env.EVALUATION_TEST_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
let backend;
let browser;
let socket;
let sequence = 0;
const pending = new Map();

async function until(check) {
  for (let attempt = 0; attempt < 150; attempt += 1) {
    if (await check()) return;
    await delay(100);
  }
  throw new Error(`Timed out: ${check}`);
}

function command(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => {pending.delete(id); reject(new Error(`CDP timed out: ${method}`));}, 20000);
    pending.set(id, {resolve, reject, timer});
    socket.send(JSON.stringify({id, method, params}));
  });
}

async function evaluate(expression) {
  const result = await command('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true, userGesture: true});
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}

async function run() {
  const env = {...process.env, PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1'};
  delete env.PYTHONHOME;
  delete env.PYTHONPATH;
  let diagnostics = '';
  let origin;
  backend = spawn(path.join(packageRoot, 'runtime/python/python.exe'), ['-I', '-B',
    path.join(packageRoot, 'evaluation-browser.py'), '--no-browser', '--port', '0', '--data', path.join(temporary, 'answers')],
  {env, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe']});
  backend.stdout.on('data', chunk => {
    diagnostics += chunk;
    origin = diagnostics.match(/Human Evaluation: (http:\/\/127\.0\.0\.1:\d+)/)?.[1];
  });
  backend.stderr.on('data', chunk => {diagnostics += chunk;});
  await until(async () => {
    if (backend.exitCode !== null) throw new Error(diagnostics);
    if (!origin) return false;
    try {return (await fetch(`${origin}/api/health`)).ok;} catch {return false;}
  });
  const profile = path.join(temporary, 'browser');
  browser = spawn(edge, ['--headless=new', '--no-first-run', '--no-default-browser-check',
    '--remote-debugging-port=0', '--remote-debugging-address=127.0.0.1', `--user-data-dir=${profile}`, 'about:blank'],
  {windowsHide: true, stdio: 'ignore'});
  const devtools = path.join(profile, 'DevToolsActivePort');
  await until(async () => fs.existsSync(devtools));
  const debugPort = fs.readFileSync(devtools, 'utf8').split('\n')[0].trim();
  const targets = await (await fetch(`http://127.0.0.1:${debugPort}/json/list`)).json();
  socket = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {socket.onopen = resolve; socket.onerror = reject;});
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    clearTimeout(request.timer);
    if (message.error) request.reject(new Error(JSON.stringify(message.error)));
    else request.resolve(message.result);
  };
  await command('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: path.join(temporary, 'downloads')});
  await command('Page.navigate', {url: origin});
  await until(() => evaluate("document.querySelectorAll('[data-study]').length === 2"));
  assert.equal(await evaluate("typeof window.neuroclawDesktop"), 'undefined');
  await evaluate("document.querySelector('[data-study=discovery-study]').click()");
  const frame = "document.querySelector('iframe').contentDocument";
  await until(() => evaluate(`Boolean(${frame}?.querySelector('#assignment')?.options.length > 1)`));
  await evaluate(`(()=>{const doc=${frame};doc.querySelector('[data-discovery-language=en]').click();})()`);
  await until(() => evaluate(`${frame}.documentElement.lang === 'en'`));
  await evaluate(`(()=>{const doc=${frame};doc.querySelector('[data-discovery-language=zh]').click();doc.querySelector('[name=code]').value='TEST-BROWSER';doc.querySelector('[name=experience]').value='3-5';doc.querySelector('#assignment').value='P01';doc.querySelector('#setup-form').requestSubmit();})()`);
  await until(() => evaluate(`${frame}.querySelector('#workspace').hidden === false`));
  await evaluate(`${frame}.querySelector('#pause').click()`);
  await until(() => evaluate("document.querySelector('#home').hidden === false"));
  await evaluate("document.querySelector('[data-study=discovery-study]').click()");
  await until(() => evaluate(`Boolean(${frame}?.querySelector('#resume-banner')?.hidden === false)`));
  await command('Page.navigate', {url: `${origin}/study?embedded=1`});
  await until(() => evaluate("Boolean(document.querySelector('#assignment')?.options.length > 1)"));
  assert.equal(await evaluate("document.querySelector('[data-view=results]').hidden"), true);
  await evaluate("document.querySelector('#participant-id').value='TEST-BROWSER';document.querySelector('#participant-experience').value='3-5';document.querySelector('#assignment').value='P01';document.querySelector('#setup-form').requestSubmit()");
  await until(() => evaluate("document.querySelector('#workbench-view').classList.contains('active')"));
  await evaluate("document.querySelector('#save-progress').click()");
  await delay(500);
  await command('Page.navigate', {url: `${origin}/study?embedded=1`});
  await until(() => evaluate("Boolean(document.querySelector('#resume-session-btn')?.offsetParent)"));
  await evaluate("window.exportResult=null;window.exportError=null;EvaluationExport.exportResults({code:'TEST-BROWSER'}).then(value=>window.exportResult=value).catch(error=>window.exportError=String(error));void 0");
  await until(() => evaluate("Boolean(document.querySelector('dialog.evaluation-export-dialog[open]'))"));
  await evaluate("document.querySelector('dialog.evaluation-export-dialog[open] .primary').click()");
  await until(() => evaluate("Boolean(window.exportResult || window.exportError)"));
  assert.equal(await evaluate('window.exportError'), null);
  assert.equal(await evaluate('window.exportResult.downloadRequested'), true);
  let exported;
  await until(async () => {
    const downloads = path.join(temporary, 'downloads');
    if (!fs.existsSync(downloads)) return false;
    exported = fs.readdirSync(downloads).find(name => name.endsWith('.json'));
    return Boolean(exported);
  });
  const payload = JSON.parse(fs.readFileSync(path.join(temporary, 'downloads', exported), 'utf8'));
  assert.equal(payload.participant_code, 'TEST-BROWSER');
  assert.notEqual(payload.human_evaluation_1.status, 'not_included');
  assert.notEqual(payload.human_evaluation_2.status, 'not_included');
  console.log(JSON.stringify({status: 'passed', browser: edge, package: packageRoot,
    bilingual: true, he1_save_resume: true, he2_save_resume: true,
    browser_json_download: true, no_electron_bridge: true, test_data: temporary}, null, 2));
}

run().catch(error => {console.error(error); process.exitCode = 1;}).finally(async () => {
  if (socket?.readyState === WebSocket.OPEN) {
    try {await command('Browser.close');} catch {}
    socket.close();
  }
  if (browser && browser.exitCode === null) browser.kill();
  if (backend && backend.exitCode === null) backend.kill();
  for (const request of pending.values()) clearTimeout(request.timer);
});
