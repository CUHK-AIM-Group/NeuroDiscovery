const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const os=require('node:os');
const {request:httpRequest}=require('node:http');
const {setTimeout:delay}=require('node:timers/promises');
const {createDemoServer,presets,limitPrompt,matchPrompt}=require('../demo/native-server.cjs');
const D=require('../demo/scenarios.js');
const {parseDelimited}=require('../../core/web/static/artifact-panel.js');
const root=path.resolve(__dirname,'../..'),staticRoot=path.join(root,'core/web/static');
async function fixture(t,timeScale=.001){
  const profile=fs.mkdtempSync(path.join(os.tmpdir(),'nd-native-api-'));
  const api=await createDemoServer({staticRoot,profile,workspace:root,timeScale});
  t.after(()=>api.close());return {...api,profile};
}
const post=(api,url,value)=>fetch(api.origin+url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
const send=(api,p,id=p.id,chatId=id)=>post(api,'/api/chat',{request_id:id,chat_id:chatId,message:p.prompt,autoresearch_mode:p.mode});
async function done(api,id){
  const start=Date.now();while(!api.runs.get(id)?.result){if(Date.now()-start>20000)throw Error('Run timeout: '+id);await delay(10);}return api.runs.get(id);
}

test('short researcher prompts share one definition, while old prompts and loop cases still work',()=>{
  const guide=fs.readFileSync(path.join(root,'desktop/DEMO_PROMPTS.md'),'utf8');
  for(const definition of D.definitions){
    const preset=presets.find(p=>p.id===definition.id);
    assert.equal(preset.prompt,definition.prompt);assert.ok(guide.includes(preset.prompt));
    assert.doesNotMatch(preset.prompt,/\p{Script=Han}/u);
    assert.doesNotMatch(preset.prompt,/GNN|metadata|模态|超参数|调参|代码|\d+\s*(条|个|轮|次)/i);
    for(const prompt of [definition.prompt,...definition.aliases]){
      assert.equal(matchPrompt(prompt).id,definition.id);
      assert.equal(matchPrompt(prompt.replace(/。/g,'！\n')).mode,preset.mode);
    }
  }
  assert.equal(matchPrompt(limitPrompt).options.fullOutcome,'limit');
  const oldFull=D.definitions.find(p=>p.id==='full').aliases.at(-1);
  assert.equal(matchPrompt(oldFull+' 请演示持续证据不足、达到 20 轮上限的情况。').options.fullOutcome,'limit');
  assert.equal(matchPrompt('请解释帕金森病的遗传风险'),undefined);
});
test('serves the exact production frontend without keys or a second interface',async t=>{
  const api=await fixture(t);
  for(const [url,file] of [['/harness','index.html'],['/static/harness.css','harness.css'],['/static/client-workbench.js','client-workbench.js']]){
    const response=await fetch(api.origin+url);assert.equal(response.status,200);assert.deepEqual(Buffer.from(await response.arrayBuffer()),fs.readFileSync(path.join(staticRoot,file)));
  }
  const env=await(await fetch(api.origin+'/api/env')).json();assert.equal(env.model,'GPT-6-Astra');assert.equal(env.api_key_required,false);
  assert.equal((await fetch(api.origin+'/api/live-provider',{headers:{origin:'https://example.com'}})).status,403);
  assert.equal((await fetch(api.origin+'/static/%2e%2e%2f%2e%2e%2fAGENTS.md')).status,403);
  assert.equal((await fetch(api.origin+'/api/neurooracle/graph/status')).status,200);
});
test('all five documented prompts run through the native contract, with stages and artifacts',async t=>{
  const api=await fixture(t);
  for(const p of presets){
    assert.equal(matchPrompt(p.prompt).mode,p.mode);assert.equal((await send(api,p)).status,200);
    const run=await done(api,p.id);assert.equal(run.status,'completed');assert.ok(run.maxDelay<=5000);
    assert.doesNotMatch(JSON.stringify({content:run.result.content,blocks:run.blocks,tools:run.tools,assets:run.assets}),/\p{Script=Han}/u,`${p.id} must be English throughout`);
    assert.doesNotMatch(JSON.stringify({content:run.result.content,blocks:run.blocks,tools:run.tools}),/演示|模拟|预设|合成|回放|玩具|API\s*key|synthetic|executed|demo_model/i);
    assert.equal(run.result.demo,true);assert.equal(run.result.scientific_validation,false);
    assert.equal(run.result.autoresearch_mode,p.mode);
    const poll=await(await fetch(api.origin+'/api/chat/runs/'+p.id)).json();assert.equal(poll.snapshot.chat_id,p.id);assert.equal(poll.status,'completed');
  }
  assert.equal(api.runs.get('idea').tools.filter(x=>x.tool.startsWith('Agent ')).length,3);
  assert.match(api.runs.get('idea').result.content,/\| Hypothesis \|/);
  for(const id of ['idea','full']){
    const run=api.runs.get(id),preview=run.blocks.find(b=>b.text.includes('**Hypothesis ranking**')).text;
    assert.match(preview,/500 candidates/);assert.match(preview,/492 candidates/);
    assert.equal(preview.split('\n').filter(line=>/^\| \d+ \/ H/.test(line)).length,8);
    assert.equal(run.assets['hypotheses.csv'],D.hypothesisCsv(D.hypothesisPool));
  }
  assert.match(api.runs.get('full').result.content,/501 candidates/);
  assert.equal(api.runs.get('full').assets['hypotheses_final.csv'],D.hypothesisCsv(D.fullHypothesisPool()));
  assert.match(api.runs.get('data').result.content,/epochs.npy/);
  assert.match(api.runs.get('experiment').result.content,/Final test AUROC/);
  assert.match(api.runs.get('full').result.content,/H6/);
  const full=api.runs.get('full'),history=JSON.parse(full.assets['loop_history.json']);
  assert.equal(history.final_test_used_for_iteration,false);
  assert.equal(history.executed,false);
  assert.deepEqual(history.rounds.map(round=>round.verdict.count),[1,2,3]);
  for(const round of history.rounds){
    const file=`round_${String(round.round).padStart(2,'0')}.csv`;
    const response=await fetch(api.origin+'/demo/artifacts/full/'+file);
    const parsed=parseDelimited(await response.text());
    assert.equal(response.status,200);assert.equal(parsed.warning,false);
    assert.deepEqual(parsed.rows.slice(1).map(row=>row.slice(1,-2)),round.comparisons);
    assert.ok(parsed.rows.slice(1).every(row=>row.at(-1)==='simulated_not_measured'&&row.at(-2)==='development'));
    const experimentAt=full.result.content.indexOf(`**Round ${round.round} results (development)**`);
    const decisionAt=full.result.content.indexOf(`**Round ${round.round} decision**`);
    assert.ok(experimentAt>=0&&experimentAt<decisionAt,'experimental results precede the feedback');
    assert.ok(full.assets['REPORT.md'].includes(round.finding));
    assert.ok(full.assets['REPORT.md'].includes(round.nextStep));
    const phases=full.tools.filter(tool=>tool.tool.startsWith(`Round ${round.round}/20`));
    assert.equal(new Set(phases.map(tool=>tool.output)).size,4,'each phase has its own feedback');
  }
  assert.equal(parseDelimited(full.assets['round_metrics.csv']).rows.length,10);
  assert.match(await(await fetch(api.origin+'/demo/artifacts/full/IDEA.md')).text(),/## H6/);
  assert.deepEqual(api.runs.get('chat').tools.map(t=>t.tool),['Client · describe_capabilities']);
});
test('round evidence is available while streaming and cancellation cannot publish later experiments',async t=>{
  const api=await fixture(t,.03);await send(api,presets[4],'partial-round');
  const start=Date.now();
  while(!api.runs.get('partial-round')?.blocks.some(block=>block.text.includes('/round_01.csv)'))){
    if(Date.now()-start>30000)throw Error('No first-round artifact link');await delay(5);
  }
  const url=api.origin+'/demo/artifacts/partial-round/';
  const first=await fetch(url+'round_01.csv');assert.equal(first.status,200);
  const bytes=await first.text();assert.equal(parseDelimited(bytes).rows.length,4);
  await post(api,'/api/chat/cancel',{request_id:'partial-round'});
  const stopped=await done(api,'partial-round');assert.equal(stopped.status,'cancelled');
  assert.equal(await(await fetch(url+'round_01.csv')).text(),bytes);
  assert.equal((await fetch(url+'round_02.csv')).status,404);
  assert.equal((await fetch(url+'results.csv')).status,404);
  const history=JSON.parse(stopped.assets['loop_history.json']);
  assert.equal(history.rounds.length,1);assert.equal(history.rounds[0].verdict.count,1);
  assert.doesNotMatch(stopped.assets['REPORT.md'],/## Final outcome/);
});

test('full CSV is downloadable as soon as its link streams, even if the run is stopped',async t=>{
  const api=await fixture(t,.08);await send(api,presets[1],'early-csv');
  const start=Date.now();
  while(!api.runs.get('early-csv')?.blocks.some(b=>b.status==='running'&&b.text.includes('/hypotheses.csv)'))){
    if(Date.now()-start>15000)throw Error('No streaming artifact link');await delay(5);
  }
  const response=await fetch(api.origin+'/demo/artifacts/early-csv/hypotheses.csv');
  assert.equal(response.status,200);assert.match(response.headers.get('content-type'),/text\/csv/);
  assert.match(response.headers.get('content-disposition'),/hypotheses.csv/);
  const bytes=Buffer.from(await response.arrayBuffer());
  assert.ok(bytes.equals(Buffer.from(D.hypothesisCsv(D.hypothesisPool))));
  await post(api,'/api/chat/cancel',{request_id:'early-csv'});
  assert.equal((await done(api,'early-csv')).status,'cancelled');
  assert.equal((await fetch(api.origin+'/demo/artifacts/early-csv/hypotheses.csv')).status,200);
  assert.equal((await fetch(api.origin+'/demo/artifacts/early-csv/IDEA.md')).status,404);
});
test('full insufficient-evidence branch reaches 20 without falsely passing',async t=>{
  const api=await fixture(t);await send(api,{...presets[4],prompt:limitPrompt},'limit');const run=await done(api,'limit');
  assert.match(run.result.content,/Round 20 decision/);assert.match(run.result.content,/20-round limit/);
  assert.doesNotMatch(JSON.stringify({content:run.result.content,blocks:run.blocks,tools:run.tools,assets:run.assets}),/\p{Script=Han}/u);
  assert.doesNotMatch(run.assets['validation.csv'],/"passed"/);
});
test('wrong scope returns guidance, identity retries do not start a second run',async t=>{
  const api=await fixture(t);const wrong={...presets[1],mode:'off'};
  await send(api,wrong,'scope');const run=await done(api,'scope');assert.match(run.result.content,/Switch/);assert.equal(run.tools.length,0);
  await send(api,wrong,'scope');assert.equal(api.runs.size,1);
  assert.equal((await send(api,presets[1],'scope')).status,409);
});
test('cancelling a queued item does not start concurrent work in the same chat',async t=>{
  const api=await fixture(t,.08);
  await send(api,presets[1],'first','shared');await send(api,presets[0],'second','shared');await send(api,presets[0],'third','shared');
  assert.equal(api.runs.get('second').status,'queued');
  await post(api,'/api/chat/cancel',{request_id:'second'});
  assert.equal(api.runs.get('third').status,'queued');
  await post(api,'/api/chat/cancel',{request_id:'first'});
  assert.equal((await done(api,'first')).status,'cancelled');
  assert.equal((await done(api,'third')).status,'completed');
  const frozen=api.runs.get('first').seq;await delay(30);assert.equal(api.runs.get('first').seq,frozen);
});
test('history persists and rejects stale revisions',async t=>{
  const api=await fixture(t);const put=revision=>fetch(api.origin+'/api/workbench/state',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision,state:{sessions:[{id:'saved'}],projects:[]}})});
  assert.equal((await put(0)).status,200);assert.equal((await put(0)).status,409);
  const loaded=await(await fetch(api.origin+'/api/workbench/state')).json();assert.equal(loaded.state.sessions[0].id,'saved');assert.equal(loaded.revision,1);
});

test('a request body split inside a multi-byte character is not corrupted',async t=>{
  const api=await fixture(t);
  // Force one chunk per byte so every CJK character is split across TCP boundaries.
  const content='训练折内标准化；时序 .1D、影像 .nii.gz；站点校准与等量剔除';
  const body=Buffer.from(JSON.stringify({revision:0,state:{sessions:[{id:'split',messages:[{role:'assistant',content}]}],projects:[]}}),'utf8');
  const response=await new Promise((resolve,reject)=>{
    const request=httpRequest(`${api.origin}/api/workbench/state`,{method:'PUT',headers:{'Content-Type':'application/json','Content-Length':body.length}},res=>{const chunks=[];res.on('data',chunk=>chunks.push(chunk));res.on('end',()=>resolve({status:res.statusCode,text:Buffer.concat(chunks).toString('utf8')}));});
    request.on('error',reject);
    let offset=0;
    const writeNext=()=>{if(offset>=body.length)return request.end();request.write(body.subarray(offset,offset+1));offset+=1;setImmediate(writeNext);};
    writeNext();
  });
  assert.equal(response.status,200);
  const stored=JSON.parse(fs.readFileSync(path.join(api.profile,'native-history.json'),'utf8'));
  assert.equal(stored.state.sessions[0].messages[0].content,content);
  assert.doesNotMatch(stored.state.sessions[0].messages[0].content,/\uFFFD/u);
});

test('English topic titles migrate known automatic titles once and preserve custom names and messages',async t=>{
  const profile=fs.mkdtempSync(path.join(os.tmpdir(),'nd-topic-titles-'));
  const user=presets[1].prompt;
  const sessions=[
    {id:'old-auto',title:'idea',autoTitlePending:false,messages:[{role:'user',content:user},{role:'assistant',content:'Retained reply'}]},
    {id:'old-chinese-auto',title:D.definitions[1].titles.zh,autoTitlePending:false,messages:[{role:'user',content:D.definitions[1].aliases[0]},{role:'assistant',content:'保留历史消息'}]},
    {id:'custom',title:'我的课题笔记',autoTitlePending:false,messages:[{role:'user',content:user}]},
    {id:'unmatched',title:'idea',messages:[{role:'user',content:'另一个研究主题'}]},
  ];
  const original=JSON.stringify({revision:7,state:{sessions,projects:[]}});
  fs.writeFileSync(path.join(profile,'native-history.json'),original);
  const api=await createDemoServer({staticRoot,profile,workspace:root});t.after(()=>api.close());
  for(const definition of D.definitions)for(const language of ['zh','en']){
    const response=await post(api,'/api/chat/title',{user:definition.prompt,language});
    assert.equal(response.status,200);assert.equal((await response.json()).title,definition.titles.en);
  }
  const fallback=await(await post(api,'/api/chat/title',{user:'请解释海马体与记忆的关系'})).json();
  assert.match(fallback.title,/海马体与记忆/);
  const long=await(await post(api,'/api/chat/title',{user:'脑网络'.repeat(100)})).json();assert.ok(Array.from(long.title).length<=28);
  assert.equal((await post(api,'/api/chat/title',{})).status,400);
  const migrated=await(await fetch(api.origin+'/api/workbench/state')).json();
  assert.equal(migrated.revision,8);
  assert.equal(migrated.state.sessions[0].title,D.definitions[1].titles.en);
  assert.deepEqual(migrated.state.sessions[0].messages,sessions[0].messages);
  assert.equal(migrated.state.sessions[1].title,D.definitions[1].titles.en);
  assert.deepEqual(migrated.state.sessions[1].messages,sessions[1].messages);
  assert.deepEqual(migrated.state.sessions.slice(2),sessions.slice(2));
  assert.equal(fs.readFileSync(path.join(profile,'native-history-before-english-titles.json'),'utf8'),original);
  await api.close();
  const reopened=await createDemoServer({staticRoot,profile,workspace:root});t.after(()=>reopened.close());
  assert.deepEqual(await(await fetch(reopened.origin+'/api/workbench/state')).json(),migrated);
});

test('graph and downloadable artifacts survive a backend restart with identical bytes',async t=>{
  const first=await fixture(t);await send(first,presets[4],'persistent-full');const run=await done(first,'persistent-full');
  const expected={};
  for(const name of Object.keys(run.assets)){
    const response=await fetch(first.origin+'/demo/artifacts/persistent-full/'+name);
    assert.equal(response.status,200);expected[name]=await response.text();
  }
  await first.close();
  const second=await createDemoServer({staticRoot,profile:first.profile,workspace:root});t.after(()=>second.close());
  assert.equal(second.runs.size,0);
  for(const [name,content] of Object.entries(expected)){
    const response=await fetch(second.origin+'/demo/artifacts/persistent-full/'+name);
    assert.equal(response.status,200,name);assert.equal(await response.text(),content,name);
  }
  for(const url of ['/demo/artifacts/unknown/evidence.svg','/demo/artifacts/persistent-full/absent.csv','/demo/artifacts/%2e%2e%5cnative-history/evidence.svg','/demo/artifacts/persistent-full/%2e%2e%5cnative-history.json']){
    assert.equal((await fetch(second.origin+url)).status,404,url);
  }
});

test('old saved messages recover their referenced graph without replaying the research',async t=>{
  const first=await fixture(t);await send(first,presets[1],'legacy-idea');const run=await done(first,'legacy-idea');
  const profile=fs.mkdtempSync(path.join(os.tmpdir(),'nd-legacy-graph-'));
  const history=JSON.stringify({revision:7,state:{sessions:[{id:'retained',messages:[{role:'assistant',...run.result}]}]}});
  fs.writeFileSync(path.join(profile,'native-history.json'),history);
  const restored=await createDemoServer({staticRoot,profile,workspace:root});t.after(()=>restored.close());
  const response=await fetch(restored.origin+'/demo/artifacts/legacy-idea/evidence.svg');
  assert.equal(response.status,200);assert.equal(response.headers.get('content-type'),'image/svg+xml');
  assert.equal(await response.text(),run.assets['evidence.svg']);
  assert.equal(restored.runs.size,0);
  assert.equal(fs.readFileSync(path.join(profile,'native-history.json'),'utf8'),history);
  assert.equal((await fetch(restored.origin+'/demo/artifacts/unreferenced/evidence.svg')).status,404);
});
