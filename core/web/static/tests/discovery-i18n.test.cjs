const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const I=require('../discovery-study-i18n.js');
const pack=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v4.json'),'utf8'));
const catalog=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_en_v1.json'),'utf8'));
I.addMessages(catalog.strings);
const shell=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
test('all public current case text and questions translate without changing data',()=>{
  const before=JSON.stringify(pack),missing=[];
  function walk(x){if(typeof x==='string'){if(/[\u3400-\u9fff]/.test(I.translate(x)))missing.push(x);}else if(x&&typeof x==='object')Object.values(x).forEach(walk);}
  walk({meta:pack.public_meta,questions:pack.questions,pre:pack.common_pre,post:pack.common_post,cards:pack.cards.map(c=>({pre:c.pre,post:c.post}))});
  assert.deepEqual(missing,[]);assert.equal(JSON.stringify(pack),before);
});
test('dynamic table labels and mixed feedback strings translate completely',()=>{
  for(const s of ['UCLA · 双相','E1 · 主要模型 · 调整 FD','COBRE · SZ（不含 SZA）','HCP-EP · 非情感性精神病','E8 · 背景 · 情感性早期精神病（非双相验证）','ADHD -0.335；双相 -0.650；SZ/SZA -0.561'])assert.doesNotMatch(I.translate(s),/[\u3400-\u9fff]/);
  assert.equal(I.translate('材料 03 / 10'),'Material 03 / 10');
  assert.equal(I.translate('无法判断（不计分）'),'Cannot judge (not scored)');
});
test('numbers, metric IDs, paper excerpts and unknown text are not rewritten',()=>{
  for(const s of ['-0.184','[−0.431, 0.227]','ROI_to_network:108:SomMot','26 / 91','94.23%',pack.cards[0].pre.references[0].recorded_sentence,'unmapped literal'])assert.equal(I.translate(s),s);
});
test('historical packs retain their meanings and have display translations too',()=>{
  const missing=[];
  function walk(x){if(typeof x==='string'){if(/[\u3400-\u9fff]/.test(I.translate(x)))missing.push(x);}else if(x&&typeof x==='object')Object.values(x).forEach(walk);}
  for(const version of [1,2,3]) {
    const old=JSON.parse(fs.readFileSync(path.join(__dirname,`../../study_materials/cs1_discovery_pilot_v${version}.json`),'utf8'));
    walk({meta:old.public_meta,questions:old.questions,pre:old.common_pre,post:old.common_post,issues:old.issues,cards:old.cards.map(c=>({pre:c.pre,post:c.post}))});
  }
  assert.deepEqual(missing,[]);
});
test('the unauthenticated English entry also translates its static introduction',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../discovery-study.html'),'utf8');
  for(const id of ['intro-lead','pack-note']) {
    const source=html.match(new RegExp(`<p[^>]*id="${id}"[^>]*>([^<]+)</p>`))[1];
    assert.doesNotMatch(I.translate(source),/[\u3400-\u9fff]/);
  }
});
test('release entry removes pilot notice without reintroducing it after loading',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../discovery-study.html'),'utf8');
  const js=fs.readFileSync(path.join(__dirname,'../discovery-study.js'),'utf8');
  assert.doesNotMatch(html,/试用|不是正式研究招募|暂不作方法间比较|首次发现|伦理、正式招募/);
  assert.doesNotMatch(html+js,/intro-notice/);
  assert.match(html,/id="version-label">v1\.0\.0<\/span>/);
  assert.doesNotMatch(js,/\$\("#version-label"\)\.textContent\s*=/);
  assert.doesNotMatch(html,/name="consent"/);
  assert.doesNotMatch(html,/主办方可读取本机答案和版本记录|清除浏览器不会删除服务器记录|删除请求需联系主办方/);
});

test('data storage notice block and consent checkbox removed per owner',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../discovery-study.html'),'utf8');
  assert.doesNotMatch(html,/数据保存与评价边界/);
  assert.doesNotMatch(html,/name="consent"/);
  assert.doesNotMatch(html,/我自愿参与本次评审/);
});
test('release errors and material revision translate in both display modes',()=>{
  for(const s of ['请完整填写评审信息。','请确认已阅读数据保存说明并自愿参与评审。',
    '已完成的评审不可改写；请导出或开始独立的新会话。',
    '本机尚未配置该评审材料包，请联系主办方。原有问卷不受影响。',
    '固定材料：v4 · 2026-09-15 · 单轮 · 10 份材料 · 自动保存']) {
    assert.doesNotMatch(I.translate(s),/[\u3400-\u9fff]|pilot/i);
  }
  assert.equal(I.translate('v1.0.0'),'v1.0.0');
});
test('loading configuration succeeds with the removed notice absent from the DOM',async()=>{
  const js=fs.readFileSync(path.join(__dirname,'../discovery-study.js'),'utf8');
  const nodes=Object.fromEntries(['#auth-form','#setup-form','#start','#intro-lead',
    '#intro-flow','#review-context','#pack-note','#resume-banner','#resume-description',
    '#version-label','#assignment-row','#assignment','input[name=code]'].map(id=>[id,{textContent:id==='#version-label'?'v1.0.0':'',hidden:false,value:''}]));
  const context={request:async()=>({meta:pack.public_meta,questions:pack.questions,
    scoring:pack.scoring,presentation:{strings:catalog.strings}}),API:'/api/studies/discovery',
    display:{addCatalog(){}},singleRound:meta=>meta.review_flow==='single_round',multiTopic:meta=>meta.material_layout==='multitopic',
    readCredentials:()=>null,$:id=>{assert(id in nodes,`Missing element ${id}`);return nodes[id];}};
  vm.createContext(context);
  vm.runInContext(js.slice(js.indexOf('async function loadConfig('),js.indexOf('function initialize(')),context);
  await context.loadConfig();
  assert.equal(nodes['#start'].disabled,false);
  assert.equal(nodes['#start'].textContent,'开始评审 →');
  assert.match(nodes['#intro-lead'].textContent,/共 60 题/);
  assert.match(nodes['#pack-note'].textContent,/10 份材料 · 自动保存/);
  assert.doesNotMatch(I.translate(nodes['#pack-note'].textContent),/[\u3400-\u9fff]|pilot/i);
  assert.equal(nodes['#version-label'].textContent,'v1.0.0');
});
test('all inline client scripts and changed study JavaScript parse',()=>{
  for(const [,attrs,code] of shell.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g))if(!attrs.includes('src=')&&!attrs.includes('application/json'))new vm.Script(code);
  new vm.Script(fs.readFileSync(path.join(__dirname,'../discovery-study.js'),'utf8'));
});
test('fresh client startup cannot show the study invitation',()=>{
  const source=shell.slice(shell.indexOf('function showCaseStudyWelcomeIfNeeded('),shell.indexOf('function dismissCaseStudyWelcome('));
  const panel={hidden:false,attributes:{},setAttribute(k,v){this.attributes[k]=v;}};
  const context={caseStudyWelcomeEl:panel};vm.createContext(context);vm.runInContext(source,context);
  context.showCaseStudyWelcomeIfNeeded();assert.equal(panel.hidden,true);assert.equal(panel.attributes['aria-hidden'],'true');
  assert.doesNotMatch(shell,/loadDesktopSettings\(\)\.finally\(showCaseStudyWelcomeIfNeeded\)/);
  assert.doesNotMatch(shell,/caseStudyWelcomeEl\.hidden\s*=\s*false/);
});
test('manual client entry opens discovery iframe once, no reload on language/theme/re-entry',()=>{
  const source=shell.slice(shell.indexOf('function openStudyWorkspace('),shell.indexOf('function resetStudyWorkspace('));
  let src='',loads=0,language='zh',synced=0;
  const frame={get src(){return src;},set src(value){src=value;loads++;},addEventListener(){}};
  const context={state:{theme:'light',textScale:1},expertStudyFrameEl:frame,studyResultsFrameEl:null,currentUiLanguage:()=>language,syncStudyTextScale(){},syncEmbeddedTheme(){synced++;}};
  vm.createContext(context);vm.runInContext(source,context);
  context.openStudyWorkspace('expert-study');assert.match(src,/^\/discovery-study\?embedded=1&lang=zh/);
  language='en';context.state.theme='dark';context.openStudyWorkspace('expert-study');context.openStudyWorkspace('expert-study');
  assert.equal(loads,1);assert.equal(synced,2);
});
test('discard resets the cached discovery frame, while save and untrusted messages do not',()=>{
  const opening=shell.slice(shell.indexOf('function openStudyWorkspace('),shell.indexOf('function closeStudyWorkspace('));
  const marker=shell.indexOf("payload?.type !== 'neurodiscovery:close-study-workspace'");
  const start=shell.lastIndexOf("window.addEventListener('message'",marker);
  const end=shell.indexOf("window.addEventListener('message'",marker);
  let handler,loads=0,closed=0;
  const frame={contentWindow:{postMessage(){}},addEventListener(){},set src(value){this.url=value;loads++;}};
  const context={
    state:{theme:'light',textScale:1},expertStudyFrameEl:frame,hypothesisRankingFrameEl:{contentWindow:{}},studyResultsFrameEl:null,
    currentUiLanguage:()=> 'en',syncStudyTextScale(){},syncEmbeddedTheme(){},closeStudyWorkspace(){closed++;},
    window:{location:{origin:'http://local.test'},addEventListener(type,callback){handler=callback;}},
  };
  vm.createContext(context);vm.runInContext(opening+shell.slice(start,end),context);
  context.openStudyWorkspace('expert-study');
  const event={origin:'http://local.test',source:frame.contentWindow,data:{type:'neurodiscovery:close-study-workspace',discarded:false}};
  handler(event);context.openStudyWorkspace('expert-study');
  assert.equal(loads,1);assert.equal(closed,1);
  handler({...event,origin:'http://untrusted.test',data:{...event.data,discarded:true}});
  handler({...event,source:{},data:{...event.data,discarded:true}});
  assert.equal(loads,1);assert.equal(closed,1);
  handler({...event,data:{...event.data,discarded:true}});
  assert.equal(frame.url,'about:blank');assert.equal(context.state.expertStudyFrameKey,'');
  context.openStudyWorkspace('expert-study');
  assert.match(frame.url,/^\/discovery-study\?/);assert.equal(loads,3);assert.equal(closed,2);
});
test('appearance messages include language for the embedded discovery questionnaire',()=>{
  const source=shell.slice(shell.indexOf('function syncEmbeddedTheme('),shell.indexOf('function syncEmbeddedThemeFrames('));
  const sent=[],frame={contentWindow:{postMessage:(payload,origin)=>sent.push({payload,origin})}};
  const context={state:{theme:'dark',textScale:1.2},window:{location:{origin:'http://local.test'}},currentUiLanguage:()=> 'en',expertStudyFrameEl:frame,neurooracleFrameEl:{}};
  vm.createContext(context);vm.runInContext(source,context);context.syncEmbeddedTheme(frame);
  assert.equal(sent[1].payload.type,'neurodiscovery:appearance');assert.equal(sent[1].payload.language,'en');assert.equal(sent[1].origin,'http://local.test');
});
test('frozen-pack de-identification placeholders are smoothed at the display layer',()=>{
  assert.equal(I.smooth('[已移除引文]示SZ额顶连接增高，方向异质'),'有研究提示SZ额顶连接增高，方向异质');
  assert.equal(I.smooth('[已移除引文]→P3'),'某文献（引文已略）→P3');
  assert.equal(I.smooth('父节点725a56e3e759 的修订记录'),'父假设 的修订记录');
  assert.equal(I.smooth('[父假设]的组成'),'该父假设的组成');
  assert.equal(I.smooth('[citation removed] shows an increase'),'an earlier study (citation omitted) shows an increase');
  assert.equal(I.smooth('普通文本'),'普通文本');
  // translate() itself must stay lossless so catalog keys keep matching.
  assert.match(I.translate('[已移除引文]示SZ额顶连接增高'),/已移除引文/);
});
test('display switch implementation does not replace forms, inputs or iframe URLs',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../discovery-study-i18n.js'),'utf8');
  assert.doesNotMatch(source,/\.innerHTML\s*=|\.outerHTML\s*=|\.value\s*=|location\.reload|\.src\s*=/);
  assert.match(source,/textarea,\[data-original\],\[data-user-content\]/);
});
test('leaving a client study preserves the live form and pauses activity timing',()=>{
  const transition=shell.slice(shell.indexOf('function setActiveView('),shell.indexOf('function renderWorkspaceSearch('));
  assert.match(transition,/previousView === 'expert-study'[\s\S]*study-visibility',visible:false/);
  assert.match(transition,/else resetStudyWorkspace\(previousView\)/);
  const source=fs.readFileSync(path.join(__dirname,'../discovery-study.js'),'utf8');
  assert.match(source,/hostVisible &&/);
  assert.match(source,/if\(!hostVisible\)void saveAll\(\)/);
  assert.match(source,/event\.origin!==location\.origin \|\| event\.source!==window\.parent/);
});
