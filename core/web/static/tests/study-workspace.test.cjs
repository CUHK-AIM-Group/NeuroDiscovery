'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const UI = require('../study-workspace.js');
const source = fs.readFileSync(path.join(__dirname, '../discovery-study.js'), 'utf8');

function fixture(search = '?embedded=1&theme=dark&textScale=1.2', inFrame = true) {
  const handlers = {}, properties = {}, panes = [{scrollTop: 42}, {scrollTop: 300}, {scrollTop: 90}];
  let focused = false;
  const info = {open: true, contains: target => target === info, querySelector: () => ({focus() {focused = true;}})};
  const doc = {
    documentElement: {dataset: {}, style: {setProperty(k, v) {properties[k] = v;}}},
    addEventListener(k, fn) {handlers[k] = fn;},
    querySelector: () => info.open ? info : null,
    querySelectorAll(selector) {assert.equal(selector, '#app, #material, .questions'); return panes;},
  };
  const win = {document: doc, location: {search}};
  win.parent = inFrame ? {} : win;
  return {doc, win, handlers, properties, panes, info, focused: () => focused};
}

test('scale uses the same safe range as the client without whole-page zoom', () => {
  for (const [value, expected] of [[undefined,1], [null,1], ['',1], ['bad',1], [Infinity,1], [.1,.8], [5,1.5], [1.2,1.2]]) {
    assert.equal(UI.appearance({scale:value}).scale, expected);
  }
  assert.equal(UI.appearance({theme:'dark'}).theme, 'dark');
  assert.equal(UI.appearance({theme:'invalid'}).theme, 'light');
  assert.doesNotMatch(source, /body\.style\.zoom/);
});

test('embedding requires a real parent, and standalone identity is retained', () => {
  for (const [search, frame, expected] of [['?embedded=1',true,'true'], ['?embedded=1',false,'false'], ['',true,'false']]) {
    const f = fixture(search, frame); UI.mount(f.win);
    assert.equal(f.doc.documentElement.dataset.embedded, expected);
  }
  const results = fixture('?embedded=1&view=results'); UI.mount(results.win);
  assert.equal(results.doc.documentElement.dataset.studyView, 'results');
});

test('appearance-only changes preserve pane position and unspecified theme/scale', () => {
  const f = fixture(), api = UI.mount(f.win);
  api.applyAppearance({scale:1.5});
  assert.equal(f.doc.documentElement.dataset.theme, 'dark');
  api.applyAppearance({theme:'light'});
  assert.equal(f.properties['--study-text-scale'], '1.5');
  assert.deepEqual(f.panes.map(p=>p.scrollTop), [42,300,90]);
  api.resetReadingPosition();
  assert.deepEqual(f.panes.map(p=>p.scrollTop), [0,0,0]);
});

test('standalone navigation is not moved into embedded scroll panes', () => {
  const f = fixture('', false); UI.mount(f.win).resetReadingPosition();
  assert.deepEqual(f.panes.map(p=>p.scrollTop), [42,300,90]);
});

test('wheel fallback moves a bounded reading pane and preserves zoom gestures', () => {
  const pane = {scrollTop:20, scrollHeight:500, clientHeight:100};
  let prevented = 0;
  const event = {deltaY:3, deltaMode:1, preventDefault(){prevented++;}};
  assert.equal(UI.wheelDelta(event, pane.clientHeight), 54);
  assert.equal(UI.scrollWithWheel(pane,event), true);
  assert.equal(pane.scrollTop,74);
  assert.equal(prevented,1);
  assert.equal(UI.scrollWithWheel(pane,{deltaY:2,deltaMode:1,ctrlKey:true}),false);
  assert.equal(pane.scrollTop,74);
  pane.scrollTop=400;
  assert.equal(UI.scrollWithWheel(pane,{deltaY:1,deltaMode:1,preventDefault(){prevented++;}}),false);
  assert.equal(prevented,1);
});

function wheelFixture() {
  const f = fixture();
  f.win.getComputedStyle = el => ({overflowY: (el && el._overflowY) || 'visible'});
  UI.mount(f.win);
  return f;
}
function wheelNode(overflowY, extra = {}) {
  return {nodeType:1, scrollTop:0, scrollHeight:800, clientHeight:200, parentElement:null, _overflowY:overflowY, ...extra};
}

test('embedded wheel delegation scrolls the pane under the cursor, including the sidebar', () => {
  const f = wheelFixture();
  const sidebar = wheelNode('auto');
  const navChild = {nodeType:1, scrollTop:0, scrollHeight:20, clientHeight:20, parentElement:sidebar};
  const section = {nodeType:1, scrollTop:0, scrollHeight:30, clientHeight:30, parentElement:navChild};
  let prevented = 0;
  f.handlers.wheel({target:section, deltaY:120, deltaMode:0, preventDefault(){prevented++;}});
  assert.equal(sidebar.scrollTop, 120);
  assert.equal(prevented, 1);
});

test('embedded wheel delegation chains to an ancestor when the inner pane is at its boundary', () => {
  const f = wheelFixture();
  const pane = wheelNode('auto');
  const inner = wheelNode('auto', {scrollTop:600, parentElement:pane});
  let prevented = 0;
  f.handlers.wheel({target:inner, deltaY:120, deltaMode:0, preventDefault(){prevented++;}});
  assert.equal(inner.scrollTop, 600);
  assert.equal(pane.scrollTop, 120);
  assert.equal(prevented, 1);
});

test('embedded wheel delegation contains overscroll when nothing can move', () => {
  const f = wheelFixture();
  const pane = wheelNode('auto');
  const inner = wheelNode('auto', {parentElement:pane});
  let prevented = 0;
  f.handlers.wheel({target:inner, deltaY:-120, deltaMode:0, preventDefault(){prevented++;}});
  assert.equal(inner.scrollTop, 0);
  assert.equal(pane.scrollTop, 0);
  assert.equal(prevented, 1);
});

test('embedded wheel delegation preserves zoom and horizontal gestures', () => {
  const f = wheelFixture();
  const pane = wheelNode('auto');
  let prevented = 0;
  f.handlers.wheel({target:pane, deltaY:120, deltaMode:0, ctrlKey:true, preventDefault(){prevented++;}});
  f.handlers.wheel({target:pane, deltaY:0, deltaX:100, deltaMode:0, preventDefault(){prevented++;}});
  assert.equal(pane.scrollTop, 0);
  assert.equal(prevented, 0);
});

test('single-round question order follows the material narrative', () => {
  const slice = source.slice(source.indexOf('const QUESTION_DISPLAY_ORDER'), source.indexOf('function questionHTML'));
  const context = {};
  vm.createContext(context);
  vm.runInContext(`${slice}; this.displayQuestions = displayQuestions;`, context);
  const packOrder = ['grounding','novelty','design','validation','value','feedback'].map(id => ({id}));
  assert.deepEqual(
    Array.from(context.displayQuestions(packOrder), q => q.id),
    ['novelty','grounding','feedback','design','validation','value'],
  );
  assert.deepEqual(
    Array.from(context.displayQuestions([{id:'x1'},{id:'design'},{id:'x2'}]), q => q.id),
    ['design','x1','x2'],
  );
});

test('material reading order is finding, origin, design, results in single-round mode', () => {
  const body = source.slice(source.indexOf('function renderMaterial('), source.indexOf('function questionHTML'));
  const single = body.slice(body.indexOf('// Single-round reading order'));
  for (const marker of ['这个发现是什么','它从何而来','实验设计','实验结果','前序假设实验与真实反馈']) {
    assert.ok(single.includes(marker), marker);
  }
  const order = ['这个发现是什么','它从何而来','实验设计 · 模型、数据与指标','实验结果 · 内部与外部验证']
    .map(marker => single.indexOf(marker));
  assert.deepEqual([...order].sort((a,b)=>a-b), order);
  // The design part shows only the shared essentials; row-level model details
  // and the shared boundary block no longer reach the single-round panel.
  assert.ok(!single.includes('逐行模型与原尺度参数'));
  assert.ok(!single.includes('共同方法、数据接触与统计边界'));
  assert.ok(single.indexOf('统计模型') < single.indexOf('internalHTML(post.internal_rows)'));
});

test('materials sidebar is user-resizable in both modes', () => {
  const page = fs.readFileSync(path.join(__dirname, '../discovery-study.html'), 'utf8');
  const embedded = fs.readFileSync(path.join(__dirname, '../study-workspace.css'), 'utf8');
  const standalone = fs.readFileSync(path.join(__dirname, '../discovery-study.css'), 'utf8');
  assert.match(page, /class="sidebar-resizer" id="sidebar-resizer" role="separator"/);
  assert.match(standalone, /grid-template-columns:var\(--sidebar-w,180px\)/);
  assert.match(embedded, /grid-template-columns: var\(--sidebar-w, 164px\)/);
  assert.match(embedded, /\.sidebar-resizer \{ display: none; \}/);
  assert.match(source, /initializeSidebarResizer\(\);/);
  assert.match(source, /localStorage\.setItem\(SIDEBAR_WIDTH_KEY/);
});

test('new material section headings have display translations', () => {
  const i18n = require('../discovery-study-i18n.js');
  for (const text of ['这个发现是什么','它从何而来 · 相关研究与前序实验的启发','实验设计 · 模型、数据与指标','实验结果 · 内部与外部验证','前序假设实验与真实反馈 · 自动迭代','调整目录栏宽度',
    '所有案例共用同一套数据、测量与模型。','数据集','测量指标','统计模型',
    '内部实验用 TCP 队列 234 人（ADHD 13 人、双相 26 人、SZ/SZA 16 人，各比较的对照组 91 人）；外部验证用三个互不相干的独立队列：UCLA 257 人、COBRE 158 人、HCP-EP 145 人。',
    '静息态功能连接（fMRI）。每个假设预先锁定一条连接：某个脑分区到某个网络、或两个网络之间连接强度的平均值。看两个读数：病例组与对照的差异大小（标准化效应），以及差异方向在重复抽样中的一致程度（联合方向频率）。',
    '普通最小二乘（OLS）回归，比较病例组与对照，调整年龄、性别和扫描站点；外部分析另调整头动。结果的稳定性用重复抽样检验：内部 512 次、外部 2,000 次。候选假设由系统在神经影像知识图谱上提出、经三个大语言模型评审角色讨论；验证刻意使用可解释的线性模型，没有使用深度学习。']) {
    assert.notEqual(i18n.translate(text), text, text);
  }
});

test('session explanation closes with Escape and restores summary focus', () => {
  const f = fixture(); UI.mount(f.win);
  f.handlers.keydown({key:'Tab'}); assert.equal(f.info.open, true);
  f.handlers.keydown({key:'Escape'}); assert.equal(f.info.open, false); assert.equal(f.focused(), true);
});

test('clicking inside the explanation keeps it open; outside closes it', () => {
  const f = fixture(); UI.mount(f.win);
  f.handlers.pointerdown({target:f.info}); assert.equal(f.info.open, true);
  f.handlers.pointerdown({target:{}}); assert.equal(f.info.open, false);
});

function bridge(embedded = true) {
  const updates = [], languages = [], handlers = {};
  const parent = {}, origin = 'http://localhost:9000';
  const context = {
    StudyWorkspace: {applyAppearance:p=>updates.push(p)},
    display: {setLanguage:(l,opts)=>languages.push([l,opts])},
    embedded, location:{origin}, window:{parent,addEventListener:(k,f)=>handlers[k]=f},
    params:new URLSearchParams(), hostVisible:true, saveAll:async()=>{},
  };
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf('function hostAppearance('),source.indexOf("$('#close-study').hidden")),context);
  updates.length = 0;
  return {context, updates, languages, handlers, parent, origin};
}

test('appearance messages accept only the current same-origin host frame', () => {
  const f = bridge(), payload = {type:'neurodiscovery:appearance',theme:'dark',language:'en',scale:1.4};
  for (const event of [{origin:'https://foreign.test',source:f.parent,data:payload}, {origin:f.origin,source:{},data:payload}, {origin:f.origin,source:f.parent,data:null}]) f.handlers.message(event);
  assert.equal(f.updates.length, 0);
  f.handlers.message({origin:f.origin,source:f.parent,data:payload});
  assert.equal(f.updates.length, 1); assert.equal(f.languages[0][0], 'en'); assert.equal(f.languages[0][1].notify, false);
});

test('standalone pages ignore host-looking messages', () => {
  const f = bridge(false);
  f.handlers.message({origin:f.origin,source:f.parent,data:{type:'neuroclaw:text-scale',scale:1.5}});
  assert.equal(f.updates.length, 0);
});

test('navigation resets reading panes only after saving, never on save failure', async () => {
  for (const failure of [true, false]) {
    const events = [];
    const context = {current:0,saveAll:async()=>{events.push('save');if(failure)throw Error('offline');},render:()=>events.push('render'),StudyWorkspace:{resetReadingPosition:()=>events.push('reset')},window:{matchMedia:()=>({matches:true}),scrollTo:opts=>{events.push('scroll');assert.equal(opts.behavior,'auto');}}};
    vm.createContext(context);
    vm.runInContext(source.slice(source.indexOf('async function navigate('),source.indexOf('function render()')),context);
    await context.navigate(2);
    assert.equal(context.current, failure?0:2);
    assert.deepEqual(events, failure?['save']:['save','render','reset','scroll']);
  }
});

test('closing a study asks to save or discard, and stays open on failure', async () => {
  async function run({active=true, choice='save', saveFails=false, deleteFails=false}) {
    let clicked, sent=0, saved=0, deleted=0, removed=false;
    const elements={
      '#close-study':{hidden:false,addEventListener:(e,fn)=>clicked=fn},
      '#workspace':{hidden:!active},
      '#close-dialog':{open:false,showModal(){this.open=true;},close(){this.open=false;},oncancel:null},
      '#close-discard':{onclick:null},
      '#close-save':{onclick:null},
    };
    const context={
      $:id=>{assert(id in elements,`unexpected ${id}`);return elements[id];},
      embedded:true,
      data:active?{session:{stage:'review'}}:null,
      saveProgress:async()=>{saved++;if(saveFails)throw Error('offline');},
      EvaluationExport:{forget:()=>{}},
      request:async()=>{deleted++;if(deleteFails)throw Error('gone');},
      sessionPath:()=>'/api/studies/discovery/sessions/s1',
      credentials:{id:'s1',token:'t'},dirty:new Set(['c']),STORAGE:'k',
      localStorage:{removeItem:()=>removed=true},
      message:()=>{},
      window:{parent:{postMessage:payload=>{
        assert.equal(payload.discarded,active && choice==='discard');
        sent++;
      }}},location:{origin:'http://localhost:9000'},
    };
    vm.createContext(context);
    vm.runInContext(source.slice(source.indexOf("$('#close-study').hidden"),source.lastIndexOf('loadConfig();')),context);
    const pending=clicked();
    await new Promise(resolve=>setTimeout(resolve,0));
    if(active){
      assert.equal(elements['#close-dialog'].open,true);
      if(choice==='stay')elements['#close-dialog'].oncancel({preventDefault(){}});
      else elements[choice==='save'?'#close-save':'#close-discard'].onclick();
    }
    await pending;
    return {sent,saved,deleted,removed,dirtySize:context.dirty.size};
  }
  // No active session: closes immediately without the dialog.
  assert.deepEqual(await run({active:false}),{sent:1,saved:0,deleted:0,removed:false,dirtySize:1});
  // Escape means stay: nothing is saved, deleted or closed.
  assert.deepEqual(await run({choice:'stay'}),{sent:0,saved:0,deleted:0,removed:false,dirtySize:1});
  // Save path persists first; a save failure keeps the page open.
  assert.deepEqual(await run({}),{sent:1,saved:1,deleted:0,removed:false,dirtySize:1});
  assert.deepEqual(await run({saveFails:true}),{sent:0,saved:1,deleted:0,removed:false,dirtySize:1});
  // Discard deletes the session and drops the local resume code; a delete
  // failure keeps the page open.
  const discarded=await run({choice:'discard'});
  assert.deepEqual([discarded.sent,discarded.deleted,discarded.removed,discarded.dirtySize],[1,1,true,0]);
  assert.deepEqual(await run({choice:'discard',deleteFails:true}),{sent:0,saved:0,deleted:1,removed:false,dirtySize:1});
});

test('scroll activity in independent reading panes still contributes to active time', () => {
  assert.match(source, /capture:name==='scroll'/);
  assert.match(source, /hostVisible && !\$\("#workspace"\)\.hidden/);
});

test('historical study URLs without a scale no longer shrink the UI to 80 percent', () => {
  const legacy = fs.readFileSync(path.join(__dirname, '../study.html'), 'utf8');
  const app = {style:{}};
  const context = {document:{querySelector:()=>app}};
  vm.createContext(context);
  vm.runInContext(legacy.slice(legacy.indexOf('function applyHostTextScale('),legacy.indexOf("    applyHostTextScale(params.get")),context);
  for (const [input, expected] of [[null,'1'],['','1'],['1.2','1.2'],['bad','1.1'],['9','1.5']]) {
    context.applyHostTextScale(input); assert.equal(app.style.zoom, expected);
  }
});

test('organizer preview shares are not offered in either expert study entry', () => {
  const legacy = fs.readFileSync(path.join(__dirname, '../study.html'), 'utf8');
  assert.match(source, /\.filter\(option=>option\.id!=="ALL"\)/);
  assert.doesNotMatch(source, /主持预览/);
  assert.match(legacy, /\.filter\(o=>o\.id!=='ALL'\)/);
  assert.doesNotMatch(legacy, /主持预览|Organizer preview/);
});

test('deleting an unfinished extension session returns to the setup form', () => {
  const legacy = fs.readFileSync(path.join(__dirname, '../study.html'), 'utf8');
  const restart = legacy.slice(legacy.indexOf("$('restart-session-btn')"), legacy.indexOf('function beginSession'));
  assert.ok(restart.length > 0);
  assert.doesNotMatch(restart, /createFreshSession|beginSession/);
  assert.match(restart, /\$\('resume-card'\)\.hidden=true/);
  assert.match(restart, /\$\('assignment'\)\.value=''/);
  assert.match(restart, /\$\('participant-id'\)\.focus\(\)/);
  assert.match(legacy, /async function createFreshSession\(\)/);
});

test('the extension study no longer pops a first-open intro dialog', () => {
  const legacy = fs.readFileSync(path.join(__dirname, '../study.html'), 'utf8');
  assert.doesNotMatch(legacy, /study-intro|INTRO_DISMISS_KEY|showStudyIntroIfNeeded/);
});

test('the extension welcome copy does not leak validation truth or score gaps', () => {
  const legacy = fs.readFileSync(path.join(__dirname, '../study.html'), 'utf8');
  assert.doesNotMatch(legacy, /每对一个经内部实验验证|exactly one per pair was internally validated/);
});

test('setup forms are grids so the start button never touches the selects', () => {
  const base = fs.readFileSync(path.join(__dirname, '../discovery-study.css'), 'utf8');
  const embedded = fs.readFileSync(path.join(__dirname, '../study-workspace.css'), 'utf8');
  assert.match(base, /#setup-form,#auth-form\{display:grid;gap:15px/);
  assert.match(embedded, /:is\(#setup-form, #auth-form\) \{ display: grid; gap: 16px/);
});

test('both expert studies share one compact type scale', () => {
  const allowed = new Set(['12', '13', '14', '15', '18', '24']);
  for (const file of ['../discovery-study.css', '../study-workspace.css', '../study.html']) {
    const text = fs.readFileSync(path.join(__dirname, file), 'utf8');
    for (const m of text.matchAll(/font-size:\s*(?:calc\()?(\d+(?:\.\d+)?)px/g)) {
      assert.ok(allowed.has(m[1]), `${file}: unexpected font size ${m[1]}px`);
    }
    for (const m of text.matchAll(/font:\s*[^;}]*?(\d+(?:\.\d+)?)px/g)) {
      assert.ok(allowed.has(m[1]), `${file}: unexpected shorthand font size ${m[1]}px`);
    }
  }
});

test('auto-matched HE2 references show the computed relation plus the disclosure caveat', () => {
  const legacy = fs.readFileSync(path.join(__dirname, '../study.html'), 'utf8');
  // The row builder carries the curation status through to the renderer.
  assert.match(legacy, /manualRelevanceStatus:String\(paper\.manual_relevance_status\|\|''\)\.trim\(\)/);
  // The disclosure-only relevance reason must not suppress the computed
  // what-it-did / relation text for auto-matched references.
  assert.match(legacy, /autoMatched=String\(paper\.manualRelevanceStatus\|\|''\)\.trim\(\)\.toLowerCase\(\)==='auto_kg_match'/);
  assert.match(legacy, /if\(curated&&!autoMatched\)return curated;/);
  // The disclosure stays visible as a small provenance caveat note.
  assert.match(legacy, /class="paper-auto-note"/);
  assert.match(legacy, /\.paper-auto-note \{ margin-top: 8px; color: #7b8796; font-size: 12px;/);
});

test('HE2 paper study details are expanded by default', () => {
  const legacy = fs.readFileSync(path.join(__dirname, '../study.html'), 'utf8');
  assert.match(legacy, /<details class="paper-details" open>/);
});
