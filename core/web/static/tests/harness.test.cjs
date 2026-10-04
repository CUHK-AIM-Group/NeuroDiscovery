const assert = require('node:assert/strict');
const { test } = require('node:test');
const { activate, mount, sync } = require('../harness.js');

function fixture(pathname = '/harness', search = '') {
  const calls = [];
  const elements = [];
  const document = {
    documentElement: { dataset: {} },
    createElement(tag) {
      const element = { tag, dataset: {}, children: [], attributes: {}, listeners: {},
        setAttribute(name, value) { this.attributes[name] = value; },
        removeAttribute(name) { delete this.attributes[name]; },
        addEventListener(name, callback) { this.listeners[name] = callback; },
        append(...children) { this.children.push(...children); },
        querySelector(selector) { return this.children.find(child => '.' + child.className === selector); },
      };
      elements.push(element);
      return element;
    },
    querySelectorAll: () => elements.filter(element => element.dataset.harnessCapability),
    querySelector(selector) {
      if (selector === '#app-sidebar') return { querySelector: () => ({ after: navigation => calls.push(['mount', navigation]) }) };
      return elements.find(element => '.' + element.className === selector) || null;
    },
  };
  const manifest = { schema_version: 1, capabilities: [
    { id: 'chat', kind: 'view', view: 'chat', label: 'Research chat' },
    { id: 'graph', kind: 'view', view: 'neurooracle', label: 'Knowledge graph' },
    { id: 'autoresearch', kind: 'configure', label: 'AutoResearch' },
    { id: 'evaluation', kind: 'view', view: 'expert-study', label: 'Human Evaluation' },
  ] };
  return { calls, elements, document, manifest, options: {
    document, location: { pathname, search }, evaluationEnabled: true,
    fetch: async url => { calls.push(['fetch', url]); return { ok: true, json: async () => manifest }; },
    adapter: { openView: view => calls.push(['view', view]), configureResearch: () => calls.push(['configure']) },
  } };
}

test('legacy interface remains untouched', async () => {
  const state = fixture('/');
  await mount(state.options);
  assert.deepEqual(state.calls, []);
  assert.deepEqual(state.document.documentElement.dataset, {});
});

test('mount reads only capabilities and never starts research', async () => {
  const state = fixture();
  await mount(state.options);
  assert.deepEqual(state.calls.map(call => call[0]), ['fetch', 'mount']);
  assert.equal(state.document.documentElement.dataset.harness, 'true');
  assert.deepEqual(state.elements.filter(element => element.dataset.harnessCapability).map(element => element.dataset.harnessCapability), ['chat', 'graph']);
});

test('removed sidebar entries retain their existing deep links', async () => {
  const research = fixture('/harness', '?capability=autoresearch');
  await mount(research.options);
  assert.deepEqual(research.calls.at(-1), ['configure']);
  const evaluation = fixture('/harness', '?capability=evaluation');
  await mount(evaluation.options);
  assert.deepEqual(evaluation.calls.at(-1), ['view', 'expert-study']);
});

test('deep links use existing views, not executable prompts', async () => {
  const state = fixture('/harness', '?capability=graph');
  await mount(state.options);
  assert.deepEqual(state.calls.at(-1), ['view', 'neurooracle']);
  const invalid = fixture('/harness', '?capability=run-shell-command');
  await mount(invalid.options);
  assert.equal(invalid.calls.length, 2);
});

test('evaluation-disabled builds cannot navigate via deep links', async () => {
  const state = fixture('/harness', '?capability=evaluation');
  state.options.evaluationEnabled = false;
  await mount(state.options);
  assert.ok(!state.elements.some(element => element.dataset.harnessCapability === 'evaluation'));
  assert.equal(state.calls.length, 2);
});

test('HTTP and schema errors do not silently activate a shell', async () => {
  const state = fixture();
  state.options.fetch = async () => ({ ok: false });
  await assert.rejects(mount(state.options), /unavailable/);
  assert.deepEqual(state.document.documentElement.dataset, {});
  state.options.fetch = async () => ({ ok: true, json: async () => ({ schema_version: 99 }) });
  await assert.rejects(mount(state.options), /Unsupported/);
});

test('unknown action cannot dispatch tools', () => {
  const state = fixture();
  activate({ kind: 'execute', id: 'autoresearch' }, state.options.adapter);
  assert.deepEqual(state.calls, []);
});

test('AutoResearch toolbar click survives the existing outside-click handler', () => {
  const fs = require('node:fs');
  const vm = require('node:vm');
  const html = fs.readFileSync(require('node:path').join(__dirname, '../index.html'), 'utf8');
  const handler = html.match(/document\.addEventListener\('click', \(e\) => \{\s+if \(!e\.target\.closest\('\[data-settings-select\]'\)\)[\s\S]*?\n    \}\);/)[0];
  let click;
  const context = {
    document: { addEventListener: (_name, callback) => { click = callback; } },
    state: { modelMenuOpen: true },
    closeAllSettingsSelects() {},
    setModelMenuOpen(open) { context.state.modelMenuOpen = open; },
  };
  vm.runInNewContext(handler, context);
  click({ target: { closest: selector => selector.includes('data-harness-capability') } });
  assert.equal(context.state.modelMenuOpen, true);
  click({ target: { closest: () => null } });
  assert.equal(context.state.modelMenuOpen, false);
});

test('desktop harness never reuses a backend without the capability endpoint', async () => {
  const fs = require('node:fs');
  const vm = require('node:vm');
  const source = fs.readFileSync(require('node:path').join(__dirname, '../../../../desktop/main.js'), 'utf8');
  const implementation = source.match(/async function requestDesktopCompatible\([^\n]*\)[\s\S]+?\n}/)[0];
  const context = {
    process: { argv: ['electron'] }, DEMO_BUILD: false,
    requestHealth: async () => true,
    requestStatusCode: async (_url, route) => route === '/api/harness' ? 404 : 200,
  };
  vm.runInNewContext(implementation, context);
  assert.equal(await context.requestDesktopCompatible('http://localhost'), false);
  context.requestStatusCode = async () => 200;
  assert.equal(await context.requestDesktopCompatible('http://localhost'), true);
  assert.ok(source.includes("if (!process.argv.includes('--legacy-ui')) desktopUiUrl.pathname = '/harness';"));
  context.process.argv = ['electron', '--legacy-ui'];
  context.requestStatusCode = async (_url, route) => route === '/api/harness' ? 404 : 200;
  assert.equal(await context.requestDesktopCompatible('http://localhost'), true);
});

test('language and active view track the runtime', async () => {
  const state = fixture();
  await mount(state.options);
  sync(state.document, { language: 'zh', view: 'neurooracle', mode: 'data' });
  const graph = state.elements.find(element => element.dataset.harnessCapability === 'graph');
  assert.equal(graph.attributes['aria-current'], 'page');
  assert.equal(graph.querySelector('.harness-label').textContent, '知识图谱');
  sync(state.document, { language: 'en', view: 'chat', mode: 'off' });
  assert.equal(graph.attributes['aria-current'], undefined);
  assert.equal(graph.querySelector('.harness-label').textContent, 'Knowledge graph');
});
