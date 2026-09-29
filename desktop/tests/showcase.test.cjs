const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const D=require('../demo/scenarios.js');
const P=require('../demo/player.js');

test('five prompt/mode contracts with derived descending scores',()=>{
  assert.deepEqual(D.definitions.map(d=>d.mode),['off','idea','data','model','end_to_end']);
  assert.equal(new Set(D.definitions.map(d=>d.prompt)).size,5);
  for(const ranked of [D.hypotheses,D.finalHypotheses])for(let i=0;i<ranked.length;i++){
    const h=ranked[i];
    const expected=.4*h.novelty+.2*h.structural+.2*h.gnn+.2*(h.statistical+h.clinical+h.methodological)/3;
    assert.ok(Math.abs(h.total-expected)<.005);
    if(i)assert.ok(ranked[i-1].total>=h.total);
  }
  assert.equal(D.hypotheses.filter(h=>h.status==='Selected').length,3);
  assert.ok(D.hypotheses.filter(h=>h.status==='Selected').every(h=>h.novelty>=70));
  const root=path.resolve(__dirname,'../..');
  for(const d of D.datasets)assert.ok(fs.existsSync(path.join(root,'skills',d.skill,'SKILL.md')));
  for(const model of ['braingnn','brainnetcnn','bnt'])assert.ok(fs.existsSync(path.join(root,'models',model)));
});
test('all scripted waits bounded; stages stay within scope',()=>{
  for(const d of D.definitions){
    for(const e of D.buildScenario(d.id).events)if(e.delay!==undefined)assert.ok(e.delay>0&&e.delay<=5000);
  }
  const kinds=id=>D.buildScenario(id).events.map(e=>e.kind);
  assert.ok(!kinds('chat').includes('training'));
  assert.ok(!kinds('idea').includes('training'));
  assert.ok(!kinds('data').includes('scores'));
  assert.ok(!kinds('experiment').includes('graph'));
  assert.equal(D.buildScenario('idea').events.filter(e=>e.name?.startsWith('Agent ')).length,3);
});

test('large pool keeps unique candidates, explicit missing reviews, and a bounded preview',()=>{
  assert.equal(D.hypothesisPool.length,500);
  assert.equal(new Set(D.hypothesisPool.map(h=>h.id)).size,500);
  assert.equal(new Set(D.hypothesisPool.map(h=>h.text)).size,500);
  const preview=D.buildScenario('idea').events.find(e=>e.kind==='scores');
  assert.equal(preview.rows.length,8);assert.equal(preview.pool.length,500);
  assert.equal(preview.artifact,'hypotheses.csv');assert.match(preview.summary,/500/);
  assert.equal(D.hypothesisPool.filter(h=>h.reviewStatus==='reviewed').length,8);
  for(const h of D.hypothesisPool.filter(h=>h.reviewStatus==='not_reviewed')){
    assert.equal(h.total,null);assert.equal(h.novelty,null);assert.equal(h.review,null);
    assert.equal(h.statistical,null);assert.equal(h.clinical,null);assert.equal(h.methodological,null);
  }
  const artifact=D.artifacts('idea')['hypotheses.csv'];
  assert.ok(artifact.startsWith('\uFEFF'));
  assert.equal(artifact.trim().split('\r\n').length,501);
  assert.match(artifact,/"not_reviewed"/);assert.doesNotMatch(artifact,/"(?:null|undefined|NaN)"/);
  assert.deepEqual(preview.rows.map(h=>h.id),D.hypothesisPool.slice(0,8).map(h=>h.id));
});

test('full preserves the original pool and exports every final candidate, including failed ones',()=>{
  const initial=D.artifacts('idea')['hypotheses.csv'];
  for(const options of [{},{fullOutcome:'limit'}]){
    const files=D.artifacts('full',options),pool=D.fullHypothesisPool(options);
    assert.equal(files['hypotheses.csv'],initial);
    assert.equal(pool.length,501);assert.equal(new Set(pool.map(h=>h.id)).size,501);
    assert.equal(files['hypotheses_final.csv'].trim().split('\r\n').length,502);
    assert.equal(pool.filter(h=>h.validationStatus==='passed').length,options.fullOutcome==='limit'?0:3);
    assert.equal(pool.find(h=>h.id==='H3').validationStatus,'insufficient');
    assert.equal(pool.filter(h=>h.reviewStatus==='not_reviewed').length,492);
  }
  const final=D.buildScenario('full').events.filter(e=>e.kind==='scores').at(-1);
  assert.equal(final.pool.length,501);assert.equal(final.rows.length,3);
  assert.equal(final.artifact,'hypotheses_final.csv');assert.match(final.summary,/501/);
});
test('full stops on distinct hypotheses or at hard cap, without forcing success',()=>{
  assert.equal(D.decideLoop({round:2,acceptedIds:['H1','H1','H1']}).stop,false);
  assert.equal(D.decideLoop({round:3,acceptedIds:['H1','H2','H6']}).reason,'target_reached');
  assert.equal(D.decideLoop({round:20,acceptedIds:[],maxRounds:100}).reason,'max_rounds');
  const success=D.buildScenario('full').events.filter(e=>e.kind==='round');
  assert.equal(success.length,3);assert.equal(success.at(-1).verdict.count,3);
  const limit=D.buildScenario('full',{fullOutcome:'limit'}).events.filter(e=>e.kind==='round');
  assert.equal(limit.length,20);assert.equal(limit.at(-1).verdict.count,0);
  assert.equal(limit.at(-1).verdict.reason,'max_rounds');
  assert.ok(!D.artifacts('full',{fullOutcome:'limit'})['validation.csv'].includes('"passed"'));
});
test('cancellation ends pending wait immediately and prevents later events',async()=>{
  const controller=new AbortController(),seen=[];
  const run=P.play([{name:'first'},{name:'second'}],{signal:controller.signal,onEvent:async e=>{seen.push(e.name);await P.wait(1400,controller.signal);}});
  setTimeout(()=>controller.abort(),10);
  await assert.rejects(run,{name:'AbortError'});assert.deepEqual(seen,['first']);
  await assert.rejects(P.wait(5001),/Invalid/);
  await assert.rejects(P.wait(NaN),/Invalid/);
});
test('exports retain simulation provenance and no future artifacts after failed loop',()=>{
  for(const id of ['idea','experiment','full'])assert.match(D.artifacts(id)['IDEA.md'],/DEMO/);
  assert.match(D.artifacts('experiment')['results.csv'],/simulated_not_measured/);
  const history=JSON.parse(D.artifacts('full')['loop_history.json']);
  assert.equal(history.scientific_validation,false);
  assert.equal(JSON.parse(D.artifacts('experiment')['config.json']).executed,false);
  assert.match(D.artifacts('full')['IDEA.md'],/## H6/);
  assert.doesNotMatch(D.artifacts('full')['IDEA.md'],/## H3/);
});
test('keyless package excludes credentials, Python backend and scientific graph files',()=>{
  const config=JSON.parse(fs.readFileSync(path.resolve(__dirname,'../electron-builder.showcase.json')));
  assert.equal(config.extraMetadata.main,'demo-main.js');
  assert.ok(config.files.includes('preload.js'));
  assert.ok(config.files.includes('demo/native-ui/**/*'));
  assert.ok(!config.files.includes('demo/**/*'));
  assert.ok(!config.files.includes('demo-preload.js'));
  assert.ok(!JSON.stringify(config).includes('runtime'));
  const main=fs.readFileSync(path.resolve(__dirname,'../demo-main.js'),'utf8');
  assert.ok(!main.includes('spawn('));assert.ok(!main.includes('loadConfig('));
  assert.match(main,/core\/web\/static/);
  assert.match(main,/native-ui/);
  assert.ok(!main.includes('loadFile('));
});
