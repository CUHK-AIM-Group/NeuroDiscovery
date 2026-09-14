/* Offline UI behavior only: no runtime, models, credentials or graph data. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const bridge = require('../oracle-workspace.js');

test('appearance validates theme/language and bounds text scale', () => {
  assert.deepEqual(bridge.appearance({theme:'dark', language:'zh', scale:1.2}), {theme:'dark', language:'zh', scale:1.2});
  assert.deepEqual(bridge.appearance({theme:'invalid', language:'de', scale:NaN}), {});
  assert.deepEqual(bridge.appearance({scale:null}), {});
  assert.equal(bridge.appearance({scale:999}).scale, 1.5);
  assert.equal(bridge.appearance({scale:-9}).scale, .8);
});

test('layout bundle loads in a browser context without CommonJS require', () => {
  const context={};
  vm.runInNewContext(fs.readFileSync(path.join(root,'vendor/oracle-layout.js'),'utf8'),context);
  assert.equal(typeof context.graphologyLibrary.FA2Layout,'function');
  assert.equal(typeof context.graphologyLibrary.layoutForceAtlas2.inferSettings,'function');
  assert.ok(context.graphologyLibrary.layoutForceAtlas2.inferSettings(3).gravity>0);
});

test('appearance messages must come from the same-origin parent', () => {
  const parent = {}, other = {}, origin = 'http://localhost:1234';
  const valid = {origin, source:parent, data:{type:'neurodiscovery:appearance'}};
  assert.equal(bridge.trustedMessage(valid, origin, parent), true);
  for (const patch of [{origin:'https://external.invalid'}, {source:other}, {data:{type:'download-graph'}}, {data:null}]) {
    assert.equal(bridge.trustedMessage({...valid,...patch}, origin, parent), false);
  }
});

function surface({narrow=false, embedded=true} = {}) {
  const elements = new Map(), callbacks = {}, applied = [];
  let focused, language='en', resizes=0;
  const element = id => {
    if (!elements.has(id)) elements.set(id, {id, attrs:{}, events:{}, open:false,
      setAttribute(key,value){this.attrs[key]=value;}, addEventListener(key,fn){this.events[key]=fn;},
      contains(target){return target===this;},
      focus(){focused=id;}, querySelector(){return element('summary');}});
    return elements.get(id);
  };
  const doc = {getElementById:element, body:{dataset:{},classList:{contains:()=>true}},
    documentElement:{dataset:{},style:{setProperty(key,value){this[key]=value;}}},
    addEventListener:(key,fn)=>{callbacks[key]=fn;}, querySelector:()=>null};
  const win = {matchMedia:()=>({matches:narrow}), addEventListener:(key,fn)=>{callbacks[key]=fn;}, dispatchEvent:()=>resizes++};
  win.parent = embedded ? {} : win;
  const ctx = {window:win, document:doc, location:{search:'?embedded=1&lang=en&theme=light',origin:'http://local'},
    URLSearchParams, requestAnimationFrame:fn=>fn(), Event:class {}, MutationObserver:class {observe(){}}, module:{exports:{}}};
  vm.runInNewContext(fs.readFileSync(path.join(root,'oracle-workspace.js'),'utf8'), ctx);
  const api=ctx.module.exports;
  api.mount({language:()=>language,onAppearance:value=>{applied.push(value); if(value.language)language=value.language;}});
  return {api, doc, win, element, callbacks, applied, get focused(){return focused;}, get resizes(){return resizes;}};
}

test('browse and inspector controls update layout, inertness and focus', () => {
  const ui=surface();
  assert.equal(ui.doc.documentElement.dataset.embedded,'true');
  assert.equal(ui.element('oracleInspector').inert,true);
  ui.api.revealInspector();
  assert.equal(ui.doc.body.dataset.oracleInspector,'open');
  assert.equal(ui.element('oracleInspector').inert,false);
  ui.element('oracleInspectorClose').events.click();
  assert.equal(ui.doc.body.dataset.oracleInspector,'closed');
  assert.equal(ui.focused,'oracleInspectorToggle');
  ui.element('oracleBrowseToggle').events.click();
  assert.equal(ui.doc.body.dataset.oracleBrowse,'closed');
  ui.element('oracleBrowseToggle').events.click();
  assert.equal(ui.focused,'ceSearch');
  assert.ok(ui.resizes>=4);
});

test('narrow screens show selected evidence instead of a hidden detail pane', () => {
  const ui=surface({narrow:true});
  ui.api.revealEvidence();
  assert.equal(ui.doc.body.dataset.oracleBrowse,'closed');
  assert.equal(ui.focused,'ceDetail');
  ui.element('oracleBrowseToggle').events.click();
  ui.api.revealInspector();
  assert.equal(ui.doc.body.dataset.oracleBrowse,'closed');
  assert.equal(ui.doc.body.dataset.oracleInspector,'open');
  assert.equal(ui.focused,'oracleInspectorClose');
});

test('Escape closes options first, then inspector, without breaking dialogs', () => {
  const ui=surface();
  ui.api.revealInspector(); ui.element('oracleGraphOptions').open=true;
  ui.callbacks.keydown({key:'Escape'});
  assert.equal(ui.element('oracleGraphOptions').open,false);
  assert.equal(ui.doc.body.dataset.oracleInspector,'open');
  ui.doc.querySelector=()=>({open:true});
  ui.callbacks.keydown({key:'Escape'});
  assert.equal(ui.doc.body.dataset.oracleInspector,'open');
  ui.doc.querySelector=()=>null;
  ui.callbacks.keydown({key:'Escape'});
  assert.equal(ui.doc.body.dataset.oracleInspector,'closed');
});

test('live parent appearance preserves browse and detail state', () => {
  const ui=surface(); ui.api.revealInspector();
  ui.element('oracleBrowseToggle').events.click();
  const event={origin:'http://local',source:ui.win.parent,data:{type:'neurodiscovery:appearance',theme:'dark',language:'zh',scale:1.3}};
  ui.callbacks.message({...event,source:{}});
  assert.equal(ui.doc.documentElement.dataset.theme,'light');
  ui.callbacks.message(event);
  assert.equal(ui.doc.documentElement.dataset.theme,'dark');
  assert.equal(ui.doc.documentElement.style['--text-scale'],1.3);
  assert.equal(ui.element('oracleBrowseLabel').textContent,'检索');
  assert.equal(ui.doc.body.dataset.oracleInspector,'open');
  assert.equal(ui.doc.body.dataset.oracleBrowse,'closed');
  assert.equal(surface({embedded:false}).doc.documentElement.dataset.embedded,'false');
});

test('graph actions stay keyboard accessible and dismiss without hiding another panel', () => {
  const ui=surface(), menu=ui.element('oracleWorkspaceOptions');
  menu.open=true;
  ui.callbacks.click({target:menu});
  assert.equal(menu.open,true);
  ui.callbacks.click({target:{}});
  assert.equal(menu.open,false);
  menu.open=true; ui.api.revealInspector();
  ui.callbacks.keydown({key:'Escape'});
  assert.equal(menu.open,false);
  assert.equal(ui.focused,'summary');
  assert.equal(ui.doc.body.dataset.oracleInspector,'open');
  assert.equal(ui.element('oracleBrowseToggle').attrs['aria-label'],'Show or hide the search panel');
});

test('the single host title identifies the selected workspace', () => {
  const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
  const code=html.match(/^  function updateHeaderContext\([^\n]*\)[\s\S]+?^  }/m)[0];
  const label={}, state={activeView:'chat'};
  const ctx={state,document:{getElementById:()=>label},tr:value=>value};
  vm.runInNewContext(code,ctx);
  for(const [view,expected] of [['chat','NeuroRuntime'],['neurooracle','NeuroOracle'],['settings','Settings'],['skills','Skills']]) {
    state.activeView=view; ctx.updateHeaderContext(); assert.equal(label.textContent,expected);
  }
  ctx.tr=value=>value==='Settings'?'设置':value; state.activeView='settings';
  ctx.updateHeaderContext(); assert.equal(label.textContent,'设置');
});

test('host reuses the graph document when theme or language changes', () => {
  const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
  const code=['syncEmbeddedTheme','openNeuroOracleViewer'].map(name=>html.match(new RegExp('^  function '+name+'\\([^\\n]*\\)[\\s\\S]+?^  }','m'))[0]).join('\n');
  const messages=[], urls=[], listeners=[];
  const frame={contentWindow:{postMessage:(message,origin)=>messages.push({message,origin})},addEventListener:(...args)=>listeners.push(args),set src(value){urls.push(value);}};
  let language='en';
  const context={state:{theme:'light',textScale:1},neurooracleFrameEl:frame,neurooraclePageEl:{classList:{add(){}}},
    renderNeuroOracleUpdateStatus(){},currentUiLanguage:()=>language,window:{location:{origin:'http://local'}}};
  vm.runInNewContext(code,context);
  context.openNeuroOracleViewer();
  context.state.theme='dark';context.state.textScale=1.4;language='zh';
  context.openNeuroOracleViewer();
  assert.equal(urls.length,1);
  assert.match(urls[0],/embedded=1/);
  assert.equal(listeners.length,1);
  assert.equal(messages.at(-1).message.language,'zh');
  assert.equal(messages.at(-1).message.scale,1.4);
  assert.equal(messages.at(-1).origin,'http://local');
});

test('selecting a sidebar conversation leaves the graph without creating a chat', () => {
  const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
  const code=html.match(/^  function setActiveSession\([^\n]*\)[\s\S]+?^  }/m)[0];
  for (const id of ['current','other']) {
    const state={sessions:[{id:'current'},{id:'other'}],activeSessionId:'current',activeView:'neurooracle'};
    const views=[];
    const ctx={state,setActiveView:view=>views.push(view),saveActiveDraft(){},saveProjects(){},saveSessions(){},renderSidebarLists(){},loadCheckpoints(){},
      getActiveSession:()=>state.sessions.find(s=>s.id===state.activeSessionId),isKnownProjectId:()=>false};
    vm.runInNewContext(code,ctx); ctx.setActiveSession(id);
    assert.deepEqual(views,['chat']);
    assert.equal(state.sessions.length,2);
    assert.equal(state.activeSessionId,id);
    ctx.setActiveSession('unknown'); assert.equal(state.activeSessionId,id);
  }
});

test('evidence rendering retains non-support, negation, raw source text and independence caveat', () => {
  const detail={innerHTML:''}, window={};
  const source=fs.readFileSync(path.join(root,'claim-evidence.js'),'utf8').replace(/\}\)\(\);\s*$/, 'window.testRender = renderDetail; window.testState = s;})();');
  vm.runInNewContext(source,{window,document:{getElementById:()=>detail}});
  const original={subject_name:'Synthetic subject',predicate:'associated_with',object_name:'Synthetic object'};
  const data={claim:original,shared_claim_id:'REL:test',original_claim_ids:['CLM:original'],reviewed_supporting_article_count:0,article_count:1,observation_count:1,
    papers:[{work_key:'fixture',bibliography:{title:'Synthetic paper'},publication_versions:[],verified:false,supports_reviewed_proposition:false,
      observations:[{claim_id:'CLM:original',observation_role:'background_assertion',proposition_support:'does_not_support',negated:true,raw_text:'Original <unsafe> & unchanged text',original_claim:original,evidence:{sample_size:0,p_value:0}}]}]};
  window.testRender(data);
  for (const phrase of ['Does not support this claim','Negated observation','Background assertion','Publication identity unverified','Paper count does not establish independent replication or consensus.','0 reviewed supporting papers','CLM:original','Original &lt;unsafe&gt; &amp; unchanged text','<dd>0</dd>']) assert.ok(detail.innerHTML.includes(phrase),phrase);
  assert.equal(detail.innerHTML.includes('<unsafe>'),false);
  window.testState.hooks={language:()=> 'zh'};
  window.testRender(data);
  for (const phrase of ['不支持该主张','否定观察','篇数不代表独立重复验证或科学共识']) assert.ok(detail.innerHTML.includes(phrase),phrase);
});
