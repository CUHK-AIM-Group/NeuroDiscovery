const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'../main.js'),'utf8');
const code=source.slice(source.indexOf('async function requestDesktopCompatible('),source.indexOf('async function findBackendPort('));
test('bundled client does not silently reuse a server with an old study pack',async()=>{
  let requested=0;
  const context={DEMO_BUILD:false,requestHealth:async()=>true,requestStatusCode:async()=>200,requestMultiTopicStudy:async()=>{requested++;return false;}};
  vm.createContext(context);vm.runInContext(code,context);
  assert.equal(await context.requestDesktopCompatible('http://127.0.0.1:7080',true),false);
  assert.equal(requested,1);
  assert.equal(await context.requestDesktopCompatible('http://127.0.0.1:7080',false),true);
  context.requestMultiTopicStudy=async()=>true;
  assert.equal(await context.requestDesktopCompatible('http://127.0.0.1:7081',true),true);
});
test('demo refuses the regular backend and never requests study configuration',async()=>{
  let demoStatus=404;
  const context={DEMO_BUILD:true,requestHealth:async()=>true,requestStatusCode:async(_url,pathname)=>{
    assert.equal(pathname,'/api/distribution/demo');return demoStatus;
  },requestMultiTopicStudy:async()=>{throw new Error('Demo must not load studies');}};
  vm.createContext(context);vm.runInContext(code,context);
  assert.equal(await context.requestDesktopCompatible('http://127.0.0.1:7080',true),false);
  demoStatus=200;
  assert.equal(await context.requestDesktopCompatible('http://127.0.0.1:7081',true),true);
});
test('the HTTP compatibility probe distinguishes readable cases from prior multi-topic releases',async()=>{
  const http=require('node:http');
  let meta={material_layout:'multitopic'};
  let workflow;
  let allocation;
  let resultsRevision;
  const server=http.createServer((_req,res)=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify({meta,evaluation_workflow_revision:workflow,experimental_results_revision:resultsRevision,assignments:{allocation_revision:allocation},scoring:{applicable_items_by_card:{'packet-01':['value']}}}));});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  try {
    const context={http};vm.createContext(context);
    vm.runInContext(source.slice(source.indexOf('function requestMultiTopicStudy('),source.indexOf('async function requestDesktopCompatible(')),context);
    const url=`http://127.0.0.1:${server.address().port}`;
    assert.equal(await context.requestMultiTopicStudy(url),false);
    meta={material_layout:'multitopic',content_revision:'case-specific-significance-v1'};
    assert.equal(await context.requestMultiTopicStudy(url),false);
    meta={material_layout:'multitopic',content_revision:'readable-case-narrative-v1'};
    assert.equal(await context.requestMultiTopicStudy(url),false);
    workflow='unified-evaluation-export-v1';
    assert.equal(await context.requestMultiTopicStudy(url),false);
    allocation='he1-topic30-v6';
    assert.equal(await context.requestMultiTopicStudy(url),false);
    resultsRevision='visible-results-v1';
    assert.equal(await context.requestMultiTopicStudy(url),false);
    allocation='he1-topic40-success-led-v7';
    assert.equal(await context.requestMultiTopicStudy(url),false);
    resultsRevision='visible-results-v2';
    assert.equal(await context.requestMultiTopicStudy(url),false);
    meta.context_revision='study-scale-and-reference-context-v1';
    assert.equal(await context.requestMultiTopicStudy(url),false);
    meta.next_research_revision='source-linked-next-research-v1';
    assert.equal(await context.requestMultiTopicStudy(url),false);
    meta.lineage_detail_revision='specific-hypothesis-lineage-v1';
    assert.equal(await context.requestMultiTopicStudy(url),false);
    resultsRevision='visible-results-v3';
    assert.equal(await context.requestMultiTopicStudy(url),true);
  } finally {await new Promise(resolve=>server.close(resolve));}
});
