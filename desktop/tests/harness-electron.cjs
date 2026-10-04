const { app, BrowserWindow, session } = require('electron');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const staticRoot = path.join(root, 'core/web/static');
const output = path.join(root, 'tmp/harness-electron-preview');
app.setPath('userData', fs.mkdtempSync(path.join(os.tmpdir(), 'neurodiscovery-ui-test-')));
app.disableHardwareAcceleration();
const requests = [];
const errors = [];
const capabilities = [
  { id: 'chat', label: 'Research chat', kind: 'view', view: 'chat' },
  { id: 'graph', label: 'Knowledge graph', kind: 'view', view: 'neurooracle' },
  { id: 'autoresearch', label: 'AutoResearch', kind: 'configure' },
  { id: 'evaluation', label: 'Human Evaluation', kind: 'view', view: 'expert-study' },
  { id: 'ranking', label: 'Hypothesis ranking', kind: 'view', view: 'hypothesis-ranking' },
  { id: 'results', label: 'Evaluation results', kind: 'view', view: 'study-results' },
];
let revision = 1;
let history = { sessions: [], projects: [] };
const server = http.createServer(async (request, response) => {
  const url = new URL(request.url, 'http://localhost');
  requests.push(`${request.method} ${url.pathname}`);
  const json = data => { response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify(data)); };
  if (url.pathname === '/api/harness') return json({ schema_version: 1, capabilities });
  if (url.pathname === '/api/workbench/state') {
    if (request.method === 'PUT') {
      let body = '';
      for await (const chunk of request) body += chunk;
      history = JSON.parse(body).state;
      revision++;
    }
    return json({ revision, state: history });
  }
  if (url.pathname === '/api/env') return json({ provider: 'fixture', model: 'Offline UI fixture', api_key_present: false, available_models: [{ provider: 'fixture', model: 'Offline UI fixture' }] });
  if (url.pathname === '/api/skills') return json({ skills: [] });
  if (url.pathname === '/api/chat/queue/__probe__') return json({ ok: true });
  if (url.pathname.startsWith('/api/chat/tasks/') && request.method === 'GET') return json({items:[]});
  if (url.pathname.startsWith('/api/chat/goals/') && request.method === 'GET') return json({items:[]});
  if (url.pathname === '/api/llm/composer-capabilities') return json({reasoning_efforts:['low','medium','high']});
  if (url.pathname === '/api/checkpoints') return json({ checkpoints: [] });
  if (url.pathname === '/api/workbench/validate-workspace') return json({ path: root });
  if (url.pathname === '/api/neurooracle/graph/status') return json({ available: false });
  if (url.pathname.startsWith('/api/')) {
    response.statusCode = 403;
    return json({ error: 'Not permitted in offline UI fixture' });
  }
  if (url.pathname.startsWith('/theme-preview/')) {
    const filename = url.pathname.slice('/theme-preview/'.length);
    if (!['explore.html', 'study.html', 'discovery-study.html', 'evaluation-home.html'].includes(filename)) {
      response.statusCode = 404;
      return response.end();
    }
    const html = fs.readFileSync(path.join(staticRoot, filename), 'utf8').replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '');
    response.setHeader('Content-Type', 'text/html');
    return response.end(html);
  }
  if (url.pathname === '/discovery-study' || url.pathname === '/study') {
    response.setHeader('Content-Type', 'text/html');
    return response.end('<!doctype html><title>Offline evaluation fixture</title><p>Authentication and scientific records are not loaded in this test.</p>');
  }
  const filename = url.pathname === '/harness' || url.pathname === '/' ? 'index.html' : url.pathname.replace(/^\/static\//, '');
  const target = path.resolve(staticRoot, filename);
  if (!target.startsWith(staticRoot + path.sep) || !fs.existsSync(target) || !fs.statSync(target).isFile()) {
    response.statusCode = 404;
    return response.end();
  }
  response.setHeader('Content-Type', ({ '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.png': 'image/png' })[path.extname(target)] || 'application/octet-stream');
  fs.createReadStream(target).pipe(response);
});

async function run() {
  await app.whenReady();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  session.defaultSession.webRequest.onBeforeRequest((details, callback) => callback({ cancel: !details.url.startsWith(origin + '/') && !details.url.startsWith('data:') }));
  const window = new BrowserWindow({ show: false, width: 1320, height: 900, webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true } });
  window.webContents.on('console-message', details => {
    const { level, message } = details;
    if (level === 'error' && !message.includes('ERR_BLOCKED_BY_CLIENT') && !message.includes('403') && !message.includes('404')) errors.push(message);
  });
  const evaluate = script => window.webContents.executeJavaScript(script);
  await window.loadURL(origin + '/harness');
  await evaluate(`new Promise((resolve, reject) => {
    const ready = () => document.querySelector('[data-harness-capability="graph"]');
    if (ready()) return resolve();
    const observer = new MutationObserver(() => { if (ready()) { observer.disconnect(); clearTimeout(timer); resolve(); } });
    observer.observe(document.body, { childList: true, subtree: true });
    const timer = setTimeout(() => { observer.disconnect(); reject(new Error('Harness mount timeout')); }, 10000);
  })`);
  fs.mkdirSync(output, { recursive: true });
  const capture = async name => {
    await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
    fs.writeFileSync(path.join(output, name + '.png'), (await window.webContents.capturePage()).toPNG());
  };
  const layout = await evaluate(`(() => {
    const rect = selector => { const value = document.querySelector(selector).getBoundingClientRect(); return { x:value.x, y:value.y, width:value.width, height:value.height }; };
    return { sidebar:rect('#app-sidebar'), header:rect('#app > header'), chat:rect('#chat-panel'), input:rect('.input-row'), brand:!!document.querySelector('#app-sidebar > .brand'), overflow:document.documentElement.scrollWidth > innerWidth };
  })()`);
  assert.equal(layout.brand, false);
  assert.equal(await evaluate(`getComputedStyle(document.querySelector('#app > header .brand')).display`), 'none');
  assert.equal(await evaluate(`document.querySelector('.harness-brand-badge')`), null);
  assert.equal(await evaluate(`document.querySelector('.harness-welcome-badge')`), null);
  assert.equal(await evaluate(`document.querySelector('.harness-welcome h1').textContent`), 'NeuroDiscovery');
  assert.equal(await evaluate(`document.querySelector('[data-parameter-current]').textContent`), 'Balanced');
  assert.equal(await evaluate(`getComputedStyle(document.querySelector('#generation-parameters-btn')).display`), 'none');
  await evaluate(`document.querySelector('#model-menu-btn').click()`);
  assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('#model-menu [data-model-panel]'),item=>item.dataset.modelPanel)`), ['models','autoresearch']);
  await evaluate(`document.querySelector('[data-model-panel="autoresearch"]').click();document.querySelector('[data-autoresearch-mode="idea"]').click();document.querySelector('#model-menu-btn').click()`);
  assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('#model-menu [data-model-panel]'),item=>item.dataset.modelPanel)`), ['models','autoresearch','generation']);
  assert.equal(await evaluate(`document.querySelector('[data-model-panel="generation"] .model-menu-name').textContent`),'Hypothesis selection');
  await capture('composer-options');
  await evaluate(`document.querySelector('[data-model-panel="generation"]').click(); document.querySelector('[data-generation-mode="novelty_first"]').click()`);
  assert.equal(await evaluate(`document.querySelector('[data-parameter-current]').textContent`), 'Novelty first');
  assert.equal(await evaluate(`HypothesisControls.normalize('novelty_first')`), 'novelty_first');
  assert.equal(await evaluate(`HypothesisControls.normalize(undefined)`), 'balanced');
  await evaluate(`document.querySelector('[data-generation-mode="balanced"]').click(); document.querySelector('[data-model-panel="root"]').click()`);
  // Data and Model runs cannot change hypothesis selection, so the route must be absent.
  for (const mode of ['data','model']) {
    await evaluate(`document.querySelector('#model-menu-btn').click()`);
    await evaluate(`document.querySelector('[data-model-panel="autoresearch"]').click();document.querySelector('[data-autoresearch-mode="${mode}"]').click();document.querySelector('#model-menu-btn').click()`);
    assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('#model-menu [data-model-panel]'),item=>item.dataset.modelPanel)`), ['models','autoresearch']);
    assert.equal(await evaluate(`Boolean(document.querySelector('#model-menu [data-model-panel="generation"]'))`), false);
  }
  await evaluate(`document.querySelector('#model-menu-btn').click()`);
  await evaluate(`document.querySelector('[data-model-panel="autoresearch"]').click();document.querySelector('[data-autoresearch-mode="idea"]').click();document.querySelector('#model-menu-btn').click()`);
  assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('#model-menu [data-model-panel]'),item=>item.dataset.modelPanel)`), ['models','autoresearch','generation']);
  assert.equal(await evaluate(`getActiveSession().noveltyMode`), 'balanced');
  assert.equal(await evaluate(`getActiveSession().reasoningEffort`),'high');
  assert.equal(await evaluate(`document.querySelector('[data-model-panel="reasoning"], [data-reasoning-effort]')`),null);
  await evaluate(`document.querySelector('#model-menu-btn').click()`);
  const composer = await evaluate(`(()=>{const rect=id=>document.querySelector(id).getBoundingClientRect();const input=rect('#msg-input'),permission=rect('.composer-permissions'),model=rect('#model-menu-btn'),send=rect('#send-btn');return {below:permission.top>=input.bottom,left:permission.right<=model.left,right:model.right<=send.left,oldPanel:document.querySelector('#message-queue')?.textContent.includes('Execution permissions and recovery') || false};})()`);
  assert.deepEqual(composer,{below:true,left:true,right:true,oldPanel:false});
  assert.equal(await evaluate(`document.querySelector('select#composer-permission')`), null);
  await evaluate(`document.querySelector('#composer-permission').click()`);
  assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('#permission-menu [role="menuitemradio"]'),item=>item.firstChild.textContent)`), ['Ask before acting','Act, ask when risky','Never ask']);
  await capture('permission-menu-light');
  await evaluate(`document.querySelector('[data-permission-mode="risk"]').click()`);
  assert.equal(await evaluate(`document.querySelector('[data-permission-label]').textContent`), 'Act, ask when risky');
  assert.equal(await evaluate(`document.querySelector('#permission-menu').hidden`), true);
  await evaluate(`document.querySelector('#composer-permission').click();document.querySelector('[data-permission-mode="never"]').click()`);
  assert.equal(await evaluate(`document.querySelector('[data-permission-label]').textContent`), 'Never ask');
  await evaluate(`document.querySelector('#composer-permission').click();document.documentElement.dataset.theme='dark'`);
  assert.equal(await evaluate(`document.querySelector('[data-permission-mode="never"]').getAttribute('aria-checked')`), 'true');
  await capture('permission-menu-dark');
  await evaluate(`document.activeElement.dispatchEvent(new KeyboardEvent('keydown',{key:'Home',bubbles:true}))`);
  assert.equal(await evaluate(`document.activeElement.dataset.permissionMode`), 'ask');
  await evaluate(`document.activeElement.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowDown',bubbles:true}))`);
  assert.equal(await evaluate(`document.activeElement.dataset.permissionMode`), 'risk');
  await evaluate(`document.activeElement.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))`);
  assert.equal(await evaluate(`document.activeElement.id`), 'composer-permission');
  assert.equal(await evaluate(`document.querySelector('#permission-menu').hidden`), true);
  await evaluate(`document.querySelector('#composer-permission').click();document.querySelector('[data-permission-mode="ask"]').click();document.documentElement.dataset.theme='light'`);
  await evaluate(`document.querySelector('#msg-input').value='/comp';document.querySelector('#msg-input').dispatchEvent(new Event('input',{bubbles:true}))`);
  assert.equal(await evaluate(`document.querySelector('.slash-command-name').textContent`), '/compact');
  await evaluate(`document.querySelector('#msg-input').value='/';document.querySelector('#msg-input').dispatchEvent(new Event('input',{bubbles:true}))`);
  assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('.slash-command-name'),item=>item.textContent).slice(0,8)`), ['/autoresearch','/data','/model','/idea','/goal','/tasks','/usage','/compact']);
  await capture('slash-commands');
  await evaluate(`document.querySelector('[data-slash-index="0"]').click()`);
  assert.equal(await evaluate(`document.querySelector('#model-trigger-meta').textContent`), 'Full');
  assert.equal(await evaluate(`document.querySelector('#msg-input').value`), '');
  await evaluate(`document.querySelector('#msg-input').value='/idea';document.querySelector('#msg-input').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('[data-slash-index="0"]').click()`);
  assert.equal(await evaluate(`document.querySelector('#model-trigger-meta').textContent`), 'Idea');
  await evaluate(`document.querySelector('#msg-input').value='/goal';document.querySelector('#msg-input').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('[data-slash-index="0"]').click()`);
  assert.equal(await evaluate(`document.querySelector('.wb-goals').open`), true);
  await capture('goal-mode-empty');
  await evaluate(`document.querySelector('.wb-goals footer button').click()`);
  assert.equal(await evaluate(`document.querySelector('.wb-dialog:not(.wb-goals) input[name="criterion"]').required`), false);
  assert.equal(await evaluate(`document.querySelector('.wb-dialog:not(.wb-goals) input[name="objective"]').required`), true);
  await evaluate(`new Promise(resolve=>{const child=document.querySelector('.wb-dialog:not(.wb-goals)');child.addEventListener('close',resolve,{once:true});child.querySelector('[data-cancel]').click();})`);
  await evaluate(`document.querySelector('.wb-goals').close()`);
  await evaluate(`document.querySelector('#msg-input').value='/tasks';document.querySelector('#msg-input').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('[data-slash-index="0"]').click()`);
  assert.equal(await evaluate(`document.querySelector('.wb-tasks').open`), true);
  await capture('scheduled-checks-empty');
  await evaluate(`document.querySelector('.wb-tasks').close()`);
  await evaluate(`document.querySelector('#msg-input').value='';document.querySelector('#msg-input').dispatchEvent(new Event('input',{bubbles:true}))`);
  assert.ok(layout.sidebar.width >= 250 && layout.sidebar.width <= 300, JSON.stringify(layout));
  assert.ok(layout.chat.x >= layout.sidebar.width - 1, JSON.stringify(layout));
  assert.ok(layout.input.width <= 1120 && layout.input.width > 900, JSON.stringify(layout));
  assert.equal(layout.overflow, false);
  await capture('light');
  await evaluate(`document.querySelector('[data-menu-toggle-id]').click()`);
  assert.equal(await evaluate(`document.activeElement.dataset.menuAction`), 'rename');
  await evaluate(`Promise.all(document.getAnimations().filter(animation => animation.effect?.getTiming().iterations !== Infinity).map(animation => animation.finished.catch(() => {})))`);
  assert.equal(await evaluate(`document.querySelector('.chat-menu.open').getAttribute('role')`), 'menu');
  assert.deepEqual(await evaluate(`(() => {const menu=document.querySelector('.chat-menu.open');const row=menu.querySelector('button');return [getComputedStyle(menu).width,getComputedStyle(menu).borderRadius,getComputedStyle(row).fontSize,getComputedStyle(row).padding];})()`), ['230px','14px','14px','9px 12px']);
  await capture('session-menu-light');
  await evaluate(`document.activeElement.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowDown',bubbles:true}))`);
  assert.equal(await evaluate(`document.activeElement.dataset.menuAction`), 'pin');
  await evaluate(`document.documentElement.dataset.theme='dark'`);
  await capture('session-menu-dark');
  window.setSize(600, 500);
  await evaluate(`window.dispatchEvent(new Event('resize'))`);
  assert.equal(await evaluate(`(() => {const rect=document.querySelector('.chat-menu.open').getBoundingClientRect();return rect.left>=0 && rect.top>=0 && rect.right<=innerWidth && rect.bottom<=innerHeight;})()`), true);
  window.setSize(1320, 900);
  await evaluate(`new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
  await evaluate(`document.querySelector('.chat-menu.open button').focus()`);
  await evaluate(`document.activeElement.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))`);
  assert.equal(await evaluate(`document.querySelector('.chat-menu.open')`), null);
  assert.equal(await evaluate(`document.activeElement.hasAttribute('data-menu-toggle-id')`), true);
  window.setSize(1320, 900);
  await evaluate(`document.documentElement.dataset.theme='light'`);
  await evaluate(`(() => {
    addAssistantBubble({id:'brand-fixture',content:'NeuroDiscovery reply preview'}, 0);
    addTypingIndicator();
    upsertExecutionIndicator('read README.md', Date.now());
    state.sessionRequests[state.activeSessionId] = {execution:{status:'running',blocks:[{call_id:'preview',status:'running',text:'Draft'}],tools:[]}};
    renderLiveExecution(); renderLiveExecution();
  })()`);
  assert.equal(await evaluate(`document.querySelectorAll('#messages .assistant-identity').length`), 4);
  assert.equal(await evaluate(`Array.from(document.querySelectorAll('#messages .assistant-identity')).every(header => header.textContent === 'NeuroDiscovery' && header.querySelector('img').getAttribute('src') === '/static/logo.png')`), true);
  await evaluate(`Promise.all(Array.from(document.querySelectorAll('.assistant-identity img'), image => image.decode()))`);
  assert.equal(await evaluate(`document.querySelectorAll('#wb-live-execution .assistant-identity').length`), 1);
  assert.equal(await evaluate(`document.querySelector('#wb-live-execution .wb-execution-status').textContent`), 'Writing a response…');
  assert.equal(await evaluate(`document.querySelector('#wb-live-execution .wb-process-label').textContent`), 'Execution process');
  assert.equal(await evaluate(`document.querySelector('#wb-live-execution .wb-process-count').textContent`), '0 tool calls');
  assert.equal(await evaluate(`document.querySelector('#wb-live-execution .wb-execution-process').open`), false);
  assert.equal(await evaluate(`document.querySelector('#typing-indicator .wb-execution-process').hidden`), false);
  assert.deepEqual(await evaluate(`(() => {const host=document.querySelector('#wb-live-execution');const status=getComputedStyle(host.querySelector('.wb-execution-status'));const process=getComputedStyle(host.querySelector('details'));const scale=parseFloat(status.getPropertyValue('--text-scale')) || 1;return [status.color,Math.round(parseFloat(status.fontSize)/scale),process.borderRadius,process.backgroundColor];})()`), ['rgb(65, 118, 230)',14,'12px','rgb(248, 250, 252)']);
  await evaluate(`document.querySelector('#welcome').hidden=true; document.querySelector('#messages .msg.assistant').hidden=true; removeTypingIndicator(); document.querySelector('#execution-indicator').remove()`);
  await capture('camellia-execution-live');
  await evaluate(`document.querySelector('#messages .msg.assistant').hidden=false; upsertExecutionIndicator('read README.md', Date.now())`);
  await evaluate(`delete state.sessionRequests[state.activeSessionId]; removeTypingIndicator(); document.querySelector('#execution-indicator').remove(); document.querySelector('#wb-live-execution').remove(); document.querySelector('#welcome').hidden=true`);
  await evaluate(`Promise.all(document.getAnimations().filter(animation => animation.effect?.getTiming().iterations !== Infinity).map(animation => animation.finished.catch(() => {})))`);
  await capture('assistant-identity-light');
  await evaluate(`(() => {
    const session=getActiveSession(); session.messages.push({id:'copy-feedback-fixture',role:'user',content:'Copy fixture only'});
    addUserBubble(session.messages.at(-1),session.messages.length-1);
    window.fixtureClipboard=[];
    Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{window.fixtureClipboard.push(text)}}});
    setMessageActionsDisabled(true);
  })()`);
  assert.equal(await evaluate(`document.querySelector('[data-msg-id="copy-feedback-fixture"][data-msg-action="copy"]').disabled`), false);
  assert.equal(await evaluate(`document.querySelector('[data-msg-id="copy-feedback-fixture"][data-msg-action="delete"]').disabled`), true);
  await evaluate(`document.querySelector('[data-msg-id="copy-feedback-fixture"][data-msg-action="copy"]').click()`);
  assert.deepEqual(await evaluate(`fixtureClipboard`), ['Copy fixture only']);
  assert.equal(await evaluate(`document.querySelector('[data-msg-id="copy-feedback-fixture"][data-msg-action="copy"]').dataset.feedback`), 'success');
  await evaluate(`Promise.all(document.getAnimations().filter(animation => animation.effect?.getTiming().iterations !== Infinity).map(animation => animation.finished.catch(() => {})))`);
  assert.equal(await evaluate(`getComputedStyle(document.querySelector('[data-msg-id="copy-feedback-fixture"]').closest('.msg-actions')).visibility`), 'visible');
  await capture('message-copy-success');
  await evaluate(`navigator.clipboard.writeText=async()=>{throw new Error('synthetic denied')}; document.querySelector('[data-msg-id="copy-feedback-fixture"][data-msg-action="copy"]').click()`);
  assert.equal(await evaluate(`document.querySelector('[data-msg-id="copy-feedback-fixture"][data-msg-action="copy"]').dataset.feedback`), 'error');
  await capture('message-copy-error');
  await evaluate(`getActiveSession().messages.pop(); document.querySelector('[data-msg-id="copy-feedback-fixture"]').closest('.msg').remove(); delete navigator.clipboard; setMessageActionsDisabled(false)`);
  await evaluate(`document.documentElement.dataset.theme='dark'`);
  await capture('assistant-identity-dark');
  await evaluate(`document.querySelector('#messages .msg.assistant').remove(); document.querySelector('#welcome').hidden=false; document.documentElement.dataset.theme='light'`);
  await evaluate(`(() => {
    const host = document.createElement('div'); host.id = 'activity-fixture';
    document.querySelector('.harness-welcome').append(host);
    window.activityFixture = {status:'running', blocks:[{call_id:'one', status:'completed', model:'Offline fixture', text:'Inspecting local files'}], tools:[{tool_id:'one', tool:'read_workspace_file', status:'running', command:'README.md', output:'<img src=x onerror=alert(1)>'}]};
    ClientWorkbench.executionView(host, window.activityFixture, {live:true, zh:true});
  })()`);
  assert.equal(await evaluate(`document.querySelector('#activity-fixture .wb-execution-status').textContent`), '正在使用工具 · read_workspace_file');
  assert.equal(await evaluate(`document.querySelectorAll('#activity-fixture details[open]').length`), 0);
  assert.equal(await evaluate(`document.querySelector('#activity-fixture img')`), null);
  await capture('activity-collapsed');
  await evaluate(`document.querySelector('#activity-fixture .wb-execution-process > summary').click(); document.querySelector('#activity-fixture [data-key="tool:one"] > summary').click()`);
  await evaluate(`activityFixture.tools[0].output += '\\nMore output'; ClientWorkbench.executionView(document.querySelector('#activity-fixture'), activityFixture, {live:true, zh:true})`);
  assert.equal(await evaluate(`document.querySelectorAll('#activity-fixture details[open]').length`), 2);
  await capture('activity-expanded');
  await evaluate(`activityFixture.status='failed'; activityFixture.tools[0].status='failed'; activityFixture.tools[0].error='Offline fixture failure'; ClientWorkbench.executionView(document.querySelector('#activity-fixture'), activityFixture, {zh:true})`);
  assert.equal(await evaluate(`document.querySelector('#activity-fixture .wb-execution-status').textContent`), '失败');
  assert.equal(await evaluate(`document.querySelectorAll('#activity-fixture details[open]').length`), 2);
  await evaluate(`document.querySelector('#activity-fixture .wb-execution-process > summary').click(); ClientWorkbench.executionView(document.querySelector('#activity-fixture'), activityFixture, {zh:true})`);
  assert.equal(await evaluate(`document.querySelector('#activity-fixture .wb-execution-process').open`), false);
  await evaluate(`document.querySelector('#activity-fixture').innerHTML=''; ClientWorkbench.executionView(document.querySelector('#activity-fixture'), activityFixture, {zh:true}); document.documentElement.dataset.theme='dark'`);
  assert.equal(await evaluate(`document.querySelectorAll('#activity-fixture details[open]').length`), 0);
  await capture('activity-history-dark');
  await evaluate(`activityFixture.status='running'; activityFixture.runtime={approval:{step_id:'approval-fixture',tool:'run_shell_command',arguments:{command:'<img src=x onerror=alert(1)>'},status:'pending'}}; ClientWorkbench.executionView(document.querySelector('#activity-fixture'),activityFixture,{live:true,zh:true})`);
  assert.equal(await evaluate(`document.querySelector('#activity-fixture .wb-execution-status').textContent`), '等待工具授权');
  assert.equal(await evaluate(`document.querySelectorAll('#activity-fixture .wb-approval button').length`), 2);
  assert.equal(await evaluate(`document.querySelector('#activity-fixture .wb-approval img')`), null);
  assert.equal(await evaluate(`document.querySelector('#activity-fixture .wb-execution-process').open`), false);
  await capture('approval-pending-dark');
  await evaluate(`activityFixture.runtime.approval.status='denied'; ClientWorkbench.executionView(document.querySelector('#activity-fixture'),activityFixture,{live:true,zh:true})`);
  assert.equal(await evaluate(`document.querySelector('#activity-fixture .wb-approval')`), null);
  await evaluate(`document.documentElement.dataset.theme='light'`);
  await evaluate(`document.querySelector('#activity-fixture').remove()`);
  assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('[data-harness-capability]'), button => button.dataset.harnessCapability)`), ['chat', 'graph']);
  await evaluate(`document.querySelector('#model-menu-btn').click(); document.querySelector('[data-model-panel="autoresearch"]').click()`);
  assert.equal(await evaluate(`document.querySelector('#model-menu').classList.contains('open')`), true);
  await evaluate(`document.querySelector('[data-autoresearch-mode="data"]').click()`);
  assert.equal(await evaluate(`document.querySelector('#model-trigger-meta').textContent`), 'Data');
  await evaluate(`document.querySelector('[data-harness-capability="graph"]').click()`);
  assert.equal(await evaluate(`document.querySelector('#neurooracle-page').hidden`), false);
  await evaluate(`window.dispatchEvent(new CustomEvent('neuroclaw:menu-action', {detail:'open-expert-study'}))`);
  assert.equal(await evaluate(`document.querySelector('#expert-study-page').hidden`), false);
  await evaluate(`document.querySelector('[data-harness-capability="chat"]').click()`);
  await evaluate(`document.querySelector('#sidebar-toggle').click()`);
  assert.equal(await evaluate(`Math.round(document.querySelector('#chat-panel').getBoundingClientRect().x)`), 0);
  await evaluate(`document.querySelector('#sidebar-toggle').click(); document.documentElement.dataset.theme='dark'; document.documentElement.lang='zh-CN'; NeuroDiscoveryHarness.sync(document,{language:'zh',view:'chat',mode:'data'})`);
  assert.equal(await evaluate(`document.querySelector('[data-harness-capability="graph"] .harness-label').textContent`), '知识图谱');
  await capture('dark');
  window.setSize(600, 780);
  await capture('narrow');
  assert.equal(await evaluate(`document.documentElement.scrollWidth > innerWidth`), false);
  await evaluate(`document.querySelector('#sidebar-toggle').click()`);
  assert.equal(await evaluate(`getComputedStyle(document.querySelector('#app-sidebar')).display === 'none'`), false);
  await capture('narrow-sidebar');
  await evaluate(`document.querySelector('#sidebar-backdrop').click()`);
  assert.equal(await evaluate(`getComputedStyle(document.querySelector('#app-sidebar')).display === 'none'`), true);
  const themes = [];
  window.setSize(1320, 900);
  for (const filename of ['explore.html', 'study.html', 'discovery-study.html', 'evaluation-home.html']) {
    await window.loadURL(`${origin}/theme-preview/${filename}`);
    if (filename === 'explore.html') {
      await evaluate(`(() => {
        document.documentElement.dataset.embedded = 'true';
        document.querySelector('#oracleViewTabs').innerHTML = '<button class="oracle-view-tab" aria-selected="true">Concept graph</button><button class="oracle-view-tab" aria-selected="false">Claim evidence</button>';
        document.querySelector('#oracleWorkspaceOptions').open = true;
        document.querySelector('#oracleGraphOptions').open = true;
        document.querySelector('.oracle-search-filters').open = true;
      })()`);
    }
    for (const theme of ['light', 'dark']) {
      const colors = await evaluate(`(() => {
        document.documentElement.dataset.theme = ${JSON.stringify(theme)};
        const styles = getComputedStyle(document.documentElement);
        return {accent: styles.getPropertyValue('--accent').trim(), background: styles.getPropertyValue('--bg').trim()};
      })()`);
      assert.equal(colors.accent, theme === 'dark' ? '#679efe' : '#4176e6', `${filename}: ${theme}`);
      assert.equal(colors.background, theme === 'dark' ? '#151517' : '#fff', `${filename}: ${theme}`);
      await evaluate(`Promise.all(document.getAnimations().filter(animation => animation.effect?.getTiming().iterations !== Infinity).map(animation => animation.finished.catch(() => {})))`);
      themes.push({ filename, theme, ...colors });
      if (filename === 'explore.html') {
        await evaluate(`document.body.classList.add('claim-evidence-active')`);
        assert.notEqual(await evaluate(`getComputedStyle(document.querySelector('#graphUpdateBtn')).display`), 'none');
        assert.notEqual(await evaluate(`getComputedStyle(document.querySelector('#graphCheckUpdateBtn')).display`), 'none');
        await evaluate(`document.body.classList.remove('claim-evidence-active')`);
        const controls = await evaluate(`(() => {
          const style = selector => getComputedStyle(document.querySelector(selector));
          return {menu:style('.oracle-workspace-menu').backgroundColor, button:style('.oracle-workspace-menu .icon-btn').backgroundColor, selected:style('.oracle-view-tab[aria-selected="true"]').backgroundColor, input:style('.domain-select').backgroundColor, task:style('#taskSelect').backgroundColor, limit:style('#limitInput').backgroundColor, scheme:style('html').colorScheme};
        })()`);
        assert.equal(controls.menu, theme === 'dark' ? 'rgb(53, 54, 56)' : 'rgb(255, 255, 255)');
        assert.equal(controls.button, 'rgba(0, 0, 0, 0)');
        assert.equal(controls.selected, theme === 'dark' ? 'rgb(67, 69, 74)' : 'rgb(229, 231, 235)');
        assert.equal(controls.input, theme === 'dark' ? 'rgb(53, 54, 56)' : 'rgb(245, 246, 247)');
        assert.equal(controls.task, controls.input);
        assert.equal(controls.limit, theme === 'dark' ? 'rgb(35, 35, 36)' : 'rgb(255, 255, 255)');
        assert.equal(controls.scheme, theme);
        assert.equal(await evaluate(`getComputedStyle(document.querySelector('#graphUpdateBtn')).color`), theme === 'dark' ? 'rgb(249, 250, 251)' : 'rgb(15, 17, 21)');
      }
      const design = await evaluate(`(() => {
        const select = document.createElement('select');
        select.innerHTML = '<option value="one">First</option><option value="two">Second</option>';
        const dialog = document.createElement('dialog');
        dialog.className = 'wb-dialog';
        dialog.innerHTML = '<button class="wb-primary">Confirm</button>';
        document.body.append(select, dialog);
        const style = getComputedStyle(select);
        const result = {appearance: style.appearance, radius: style.borderRadius,
          pickerRadius: getComputedStyle(select, '::picker(select)').borderRadius,
          dialogRadius: getComputedStyle(dialog).borderRadius,
          primary: getComputedStyle(dialog.querySelector('button')).backgroundColor,
          primaryRadius: getComputedStyle(dialog.querySelector('button')).borderRadius};
        select.value = 'two';
        result.value = select.value;
        select.remove(); dialog.remove(); return result;
      })()`);
      assert.equal(design.appearance, 'base-select', filename);
      assert.equal(design.radius, '18px', filename);
      assert.equal(design.pickerRadius, '14px', filename);
      assert.equal(design.dialogRadius, '24px', filename);
      assert.equal(design.primary, 'rgb(65, 118, 230)', filename);
      assert.equal(design.primaryRadius, '18px', filename);
      assert.equal(design.value, 'two');
      await capture(`${filename.replace('.html', '')}-${theme}`);
      if (filename === 'explore.html') {
        const previousValue = await evaluate(`document.querySelector('#depthSelect').value`);
        await window.webContents.executeJavaScript(`document.querySelector('#depthSelect').showPicker()`, true);
        assert.equal(await evaluate(`document.querySelector('#depthSelect').matches(':open')`), true);
        await capture(`graph-select-${theme}`);
        window.webContents.sendInputEvent({type:'keyDown',keyCode:'Escape'});
        window.webContents.sendInputEvent({type:'keyUp',keyCode:'Escape'});
        await evaluate('new Promise(resolve => requestAnimationFrame(resolve))');
        assert.equal(await evaluate(`document.querySelector('#depthSelect').matches(':open')`), false);
        assert.equal(await evaluate(`document.querySelector('#depthSelect').value`), previousValue);
      }
    }
  }
  assert.ok(!requests.some(request => /POST \/api\/chat|graph\/download|graph\/update/.test(request)), requests.join('\n'));
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(output, 'verification.json'), JSON.stringify({ status: 'passed', layout, themes, requests, errors, electron: process.versions.electron, scope: 'Hidden Electron renderer, synthetic APIs; real secondary-page markup/CSS with scripts removed; no real backend, providers, study data or graph' }, null, 2));
  window.destroy();
  console.log('Electron harness verification passed; screenshots: ' + output);
}

run().then(() => { server.close(); app.exit(0); }).catch(error => { console.error(error); server.close(); app.exit(1); });
