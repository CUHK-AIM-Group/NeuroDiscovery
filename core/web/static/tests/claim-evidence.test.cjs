/* User-facing installation errors, exercised without providers or graph data. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function mountError(code, language = 'en', successful = false) {
  const elements = new Map();
  function element(id = '') {
    return {id, hidden:false, innerHTML:'', textContent:'', value:'2',
      lastElementChild:{textContent:''}, classList:{toggle(){}},
      setAttribute(){}, addEventListener(){}, focus(){},
      append(child){elements.set(child.id, child);}, querySelectorAll(){return [];}};
  }
  const get = id => {
    if (!elements.has(id)) elements.set(id, element(id));
    return elements.get(id);
  };
  const document = {getElementById:get, createElement:()=>element(),
    body:element('body'), documentElement:element('html'), addEventListener(){}};
  const context = {window:{}, document, URL, URLSearchParams,
    location:{origin:'http://local',href:'http://local/explore'}, history:{replaceState(){}},
    MutationObserver:class {observe(){}}, clearTimeout, setTimeout,
    fetch:async url => successful
      ? {ok:true,json:async()=>String(url).includes('/evidence-status')
        ? {configured:true,available:true,source:'formal_current_graph',graph_revision:'formal',index_ready:true}
        : {source:'formal_current_graph',graph_revision:'formal',total:1,claims:[{
          shared_claim_id:'CLM:0',claim:{subject_name:'Hippocampus',predicate:'predicts',object_name:'Memory'},
          article_count:1,reviewed_supporting_article_count:0}]} }
      : {ok:false,status:503,statusText:'Unavailable',json:async()=>({code,error:'missing source'})}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../claim-evidence.js'),'utf8'), context);
  await context.window.ClaimEvidenceExplorer.mount({language:()=>language,legacyStart(){}});
  await new Promise(resolve => setImmediate(resolve));
  return get;
}

test('missing evidence explains installation and does not offer a futile retry', async()=>{
  for (const language of ['en','zh']) {
    const get = await mountError('evidence_not_installed', language);
    assert.match(get('ceDetail').innerHTML, language==='en' ? /not installed/ : /尚未安装/);
    assert.match(get('ceDetail').innerHTML, language==='en' ? /matching graph and evidence package/ : /版本匹配/);
    assert.doesNotMatch(get('ceDetail').innerHTML, /data-ce-retry/);
    assert.equal(get('cePrevious').disabled, true);
  }
});

test('temporary evidence failures retain the retry action', async()=>{
  const get = await mountError('evidence_unavailable');
  assert.match(get('ceDetail').innerHTML, /temporarily unavailable/);
  assert.match(get('ceDetail').innerHTML, /data-ce-retry/);
});

test('formal KG shows original claims without implying reviewed shared support', async()=>{
  for (const language of ['en','zh']) {
    const get = await mountError(null, language, true);
    assert.equal(get('ceMinimum').value, '0');
    assert.match(get('ceHeading').textContent, language === 'en' ? /Extracted claims/ : /提取声明/);
    assert.match(get('ceResults').innerHTML, language === 'en' ? /Original extraction/ : /原始提取记录/);
    assert.doesNotMatch(get('ceResults').innerHTML, /ce-support|supporting papers|获审支持论文/);
    assert.match(get('ceDetail').innerHTML, language === 'en' ? /formal graph/ : /正式图谱/);
    assert.equal(get('claimEvidenceToggle').hidden, false);
  }
});
