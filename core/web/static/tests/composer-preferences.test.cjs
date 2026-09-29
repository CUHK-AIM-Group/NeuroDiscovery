const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8');
const slice = (start, end) => html.slice(html.indexOf(start), html.indexOf(end));

function fixture() {
  const stored = new Map();
  const context = {
    state: {sessions: [], activeSessionId: null, projects: [], availableModels: [], modelMenuPanel: 'root'},
    window: {},
    AUTO_RESEARCH_MODES: ['off','idea','data','model','end-to-end'], AUTO_RESEARCH_MODE_OFF: 'off',
    SESSION_SCHEMA_VERSION: 2,
    localStorage: {setItem: (key, value) => stored.set(key, value)}, SESSION_KEY: 'sessions',
    getActiveSession() {return context.state.sessions.find(session => session.id === context.state.activeSessionId);},
    saveActiveDraft(){}, isKnownProjectId: value => !!value, nowISO: () => 'now', tr: value => value,
    saveProjects(){},renderSidebarLists(){},renderActiveSession(){},syncModelTrigger(){},loadCheckpoints(){},
    clearPendingFiles(){},removeTypingIndicator(){},removeExecutionIndicator(){},
    findReusableEmptySession: () => context.reusable,
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../hypothesis-controls.js'), 'utf8'), context);
  vm.runInContext(slice('let workbenchHistory = null;', 'function normalizeSessionState('), context);
  vm.runInContext(slice('function createSession(', 'function ensureSession('), context);
  vm.runInContext(slice('function startFreshSession(', 'function startFreshHomeSession('), context);
  return {context, stored};
}

test('legacy novelty modes migrate without reinterpreting current selections', () => {
  const {context} = fixture();
  const controls = context.window.HypothesisControls;
  assert.equal(controls.migrateStored('strict'), 'novelty_first');
  assert.equal(controls.migrateStored('novelty_first'), 'balanced');
  assert.equal(controls.migrateStored('weighted'), 'balanced');
  assert.equal(controls.migrateStored('balanced'), 'balanced');
  assert.equal(controls.normalize('novelty_first'), 'novelty_first');
  assert.equal(controls.migrateStored('toString'), 'balanced');
});

test('new chats inherit last explicit controls, not resets or an older viewed chat', () => {
  const {context, stored} = fixture();
  const chosen = {researchMode:'idea',permissionMode:'never',reasoningEffort:'high',noveltyMode:'novelty_first',repairLimit:1};
  context.rememberComposerSettings(chosen);
  const first = context.createSession('New Chat');
  for (const [key,value] of Object.entries(chosen)) assert.equal(first[key], value);
  first.permissionMode = 'ask';
  const next = context.createSession('New Chat');
  assert.equal(next.permissionMode, 'never');
  assert.equal(context.state.autoResearchMode, 'idea');
  const persisted = JSON.parse(stored.get('sessions'));
  assert.deepEqual(persisted.composerDefaults, chosen);
  const reboot = fixture().context;
  reboot.state.composerDefaults = persisted.composerDefaults;
  assert.equal(reboot.createSession('New Chat').reasoningEffort, 'high');
});

test('legacy active session migrates, invalid settings use safe defaults', () => {
  const {context} = fixture();
  context.state.sessions = [{id:'old',researchMode:'data',permissionMode:'risk',noveltyMode:'strict'}];
  context.state.activeSessionId = 'old';
  assert.equal(context.createSession('New Chat').researchMode, 'data');
  const safe = context.composerSettings({permissionMode:'invalid',researchMode:'invalid'});
  assert.equal(safe.permissionMode, 'ask');
  assert.equal(safe.researchMode, 'off');
  assert.equal(safe.reasoningEffort, 'high');
  assert.equal(context.composerSettings({reasoningEffort:'low'}).reasoningEffort, 'high');
  vm.runInContext(slice('function normalizeSessionState(', 'let hypothesisControls = null;'), context);
  assert.equal(context.normalizeSessionState({reasoningEffort:'low',autoTitlePending:false}).reasoningEffort, 'high');
  // A value that is not a current mode name is not silently trusted.
  assert.equal(context.composerSettings({noveltyMode:'strict'}).noveltyMode, 'balanced');
});

test('reused empty chats inherit preferences without modifying other conversations', () => {
  const {context} = fixture();
  const old = {id:'old',researchMode:'off',permissionMode:'ask'};
  const empty = {id:'empty',researchMode:'off',permissionMode:'ask'};
  context.state.sessions = [old,empty];context.state.activeSessionId = 'old';context.reusable = empty;
  context.isCompletelyEmptySessionForProject = session => session === empty;
  context.rememberComposerSettings({researchMode:'model',permissionMode:'risk',reasoningEffort:'high',noveltyMode:'novelty_first'});
  context.startFreshSession();
  assert.equal(empty.researchMode,'model');
  assert.equal(empty.permissionMode,'risk');
  assert.equal(empty.noveltyMode,'novelty_first');
  assert.equal(old.researchMode,'off');
  assert.equal(old.permissionMode,'ask');
});

test('generation route exists only for idea and full modes; disabling does not erase strategy', () => {
  const {context} = fixture();
  Object.assign(context, {
    modelMenuEl: {innerHTML:''}, sortedAvailableModels: () => [], currentModelChoice: () => ({}),
    escapeHtml: value => value, compactModelName: () => 'model', uiText: value => value,
    isAutoResearchActive: () => context.state.autoResearchMode !== 'off',
    supportsHypothesisSelection: () => ['idea','end-to-end'].includes(context.state.autoResearchMode),
    autoResearchDescription: () => '', autoResearchIcon: () => '', getHypothesisMode: () => 'balanced',
  });
  vm.runInContext(slice('function renderModelMenu(', 'async function composerReasoningEffort('), context);
  const showsSelection = mode => ['idea','end-to-end'].includes(mode);
  for (const mode of ['off','idea','data','model','end-to-end']) {
    context.state.autoResearchMode = mode;
    context.state.modelMenuPanel = 'root';
    context.renderModelMenu();
    assert.equal(context.modelMenuEl.innerHTML.includes('data-model-panel="generation"'), showsSelection(mode));
    assert.doesNotMatch(context.modelMenuEl.innerHTML,/data-model-panel="reasoning"|Thinking intensity|Generation parameters/);
    assert.equal(/Hypothesis selection/.test(context.modelMenuEl.innerHTML), showsSelection(mode));
  }
  // Leaving the idea/full scope closes the selection panel instead of showing it.
  for (const mode of ['off','data','model']) {
    context.state.autoResearchMode = mode;context.state.modelMenuPanel = 'generation';
    context.renderModelMenu();
    assert.equal(context.state.modelMenuPanel, 'root');
  }
  context.state.autoResearchMode = 'off';context.state.modelMenuPanel = 'generation';
  context.renderModelMenu();
  assert.equal(context.state.modelMenuPanel, 'root');
  assert.equal(context.getHypothesisMode(), 'balanced');
  context.state.modelMenuPanel = 'reasoning';context.renderModelMenu();
  assert.equal(context.state.modelMenuPanel, 'root');
});

test('model is remembered only after a successful switch; removed models are not restored', async () => {
  const {context} = fixture();
  Object.assign(context, {renderModelOptions(){},setStatus(){},modelLabel:{},syncActiveRequestUi(){},showError(){}});
  vm.runInContext(slice('async function switchModel(', 'function setSendButtonMode('), context);
  context.fetch = async () => ({ok:true,json:async () => ({provider:'test',model:'chosen'})});
  await context.switchModel('test','chosen');
  assert.equal(context.state.lastModelChoice.model,'chosen');
  context.fetch = async () => ({ok:false,json:async () => ({message:'rejected'})});
  await assert.rejects(context.switchModel('test','bad'));
  assert.equal(context.state.lastModelChoice.model,'chosen');
  vm.runInContext(slice('async function loadEnv(', '// ── Checkpoints'), context);
  const switches = [];
  context.switchModel = async (provider, model) => switches.push([provider, model]);
  context.fetch = async () => ({json:async () => ({provider:'test',model:'default',available_models:[{provider:'test',model:'chosen'}]})});
  await context.loadEnv();
  assert.deepEqual(switches, [['test','chosen']]);
  switches.length = 0;
  context.fetch = async () => ({json:async () => ({provider:'test',model:'default',available_models:[]})});
  await context.loadEnv();
  assert.deepEqual(switches, []);
});
