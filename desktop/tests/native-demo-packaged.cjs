// Exercise the actual executable without provider credentials or Python.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const net=require('node:net');
const crypto=require('node:crypto');
const {spawn}=require('node:child_process');
const {setTimeout:delay}=require('node:timers/promises');
const asar=require('@electron/asar');
const {presets}=require('../demo/native-server.cjs');
const desktop=path.resolve(__dirname,'..'),root=path.dirname(desktop),dist=path.join(desktop,'dist-demo-native');
const output=path.join(root,'tmp/demo-native-check');
let child,ws,seq=0;const checks=[];
async function main(){
  const archive=path.join(dist,'win-unpacked/resources/app.asar');
  const files=asar.listPackage(archive).map(p=>p.replaceAll('\\','/').replace(/^\//,''));
  assert.ok(!files.some(f=>/\.env|credentials|core\/web\/server|knowledge_graph|demo\/(app.js|index.html|demo.css|player.js)/.test(f)));
  for(const file of ['demo-main.js','preload.js','demo/native-server.cjs','demo/scenarios.js'])assert.ok(asar.extractFile(archive,file).equals(fs.readFileSync(path.join(desktop,file))),'stale: '+file);
  const manifest=JSON.parse(asar.extractFile(archive,'demo/NATIVE_UI.json'));
  for(const file of manifest.files){
    const actual=asar.extractFile(archive,path.join('demo','native-ui',file.path)),source=fs.readFileSync(path.join(root,'core/web/static',file.path));
    assert.ok(actual.equals(source),'production UI drift: '+file.path);
    assert.equal(crypto.createHash('sha256').update(actual).digest('hex'),file.sha256);
  }
  assert.equal(JSON.parse(asar.extractFile(archive,'package.json')).distribution,'native-ui-demo');
  checks.push(`${manifest.files.length} production frontend assets byte-identical; retired UI excluded`);
  const probe=net.createServer();await new Promise(r=>probe.listen(0,'127.0.0.1',r));const port=probe.address().port;await new Promise(r=>probe.close(r));
  const portable=process.argv.includes('--portable');
  const executable=path.join(dist,portable?'NeuroDiscovery-Native-Demo-1.0.0-Portable-x64.exe':'win-unpacked/NeuroDiscovery.exe');
  const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>/^(path|systemroot|windir|comspec|temp|tmp|userprofile|appdata|localappdata|programdata|programfiles|programfiles\(x86\)|commonprogramfiles|systemdrive|number_of_processors|processor_architecture)$/i.test(k)));
  child=spawn(executable,['--demo-smoke','--remote-debugging-port='+port],{env,windowsHide:true,stdio:'ignore'});
  let launchError;child.on('error',error=>{launchError=error;});
  let target;
  for(let i=0;i<180;i++){
    if(launchError)throw launchError;
    try{target=(await(await fetch('http://127.0.0.1:'+port+'/json')).json()).find(t=>t.type==='page'&&t.url.endsWith('/harness'));}catch{}
    if(target)break;await delay(300);
  }
  assert.ok(target,'Native executable did not load');
  ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
  const pending=new Map();
  ws.addEventListener('message',({data})=>{const message=JSON.parse(data),entry=pending.get(message.id);if(entry){pending.delete(message.id);clearTimeout(entry.timer);message.error?entry.reject(Error(JSON.stringify(message.error))):entry.resolve(message.result);}});
  const call=(method,params)=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout: '+method));},8000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});
  const js=async expression=>{const response=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(response.exceptionDetails)throw Error(JSON.stringify(response.exceptionDetails));return response.result.value;};
  async function until(expression,timeout=100000){const start=Date.now();while(!await js(expression)){if(Date.now()-start>timeout)throw Error('Timed out: '+expression);await delay(80);}}
  await until(`typeof state!=='undefined'&&state.settingsLoaded&&state.activeModel==='GPT-6-Astra'&&document.querySelector('[data-harness-capability]')`);
  assert.equal(await js('state.llmConnectionStatus.apiKeyRequired'),false);assert.equal(await js('typeof window.marked.parse'),'function');
  assert.equal(await js('getActiveSession().reasoningEffort'),'high');
  assert.equal(await js('document.documentElement.lang'),'en');
  await js(`document.querySelector('#model-menu-btn').click()`);
  assert.equal(await js(`Boolean(document.querySelector('[data-model-panel="reasoning"], [data-reasoning-effort]'))`),false);
  await js(`document.querySelector('#model-menu-btn').click()`);
  checks.push('Reasoning defaults to high without a menu control');
  async function start(p){
    await js(`startNewChat();document.querySelector('#model-menu-btn').click();document.querySelector('[data-model-panel="autoresearch"]').click();document.querySelector('[data-autoresearch-mode="${p.mode}"]').click();document.querySelector('#msg-input').value=${JSON.stringify(p.prompt)};document.querySelector('#msg-input').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('#send-btn').click();`);
  }
  await start(presets[0]);
  await until(`Boolean(document.querySelector('#wb-live-execution .bubble')?.textContent)`);
  const before=await js(`document.querySelector('#wb-live-execution .bubble').textContent.length`);
  await until(`(document.querySelector('#wb-live-execution .bubble')?.textContent.length||0)>${before}`);
  await until(`!isSessionWaiting()&&getActiveSession().messages.some(m=>m.role==='assistant')`);
  assert.match(await js(`getActiveSession().messages.at(-1).content`),/NeuroOracle/);
  assert.doesNotMatch(await js('document.body.innerText'),/\p{Script=Han}/u);
  assert.doesNotMatch(await js(`document.querySelector('#messages').textContent`),/演示|模拟|预设|合成|回放|玩具|API\s*key/i);
  checks.push('Normal-speed native chat streams without a key');console.log('PASS packaged chat '+(portable?'portable':'unpacked'));
  if(!portable){
    await start(presets[4]);
    await until(`!isSessionWaiting()&&getActiveSession().messages.some(m=>m.role==='assistant')`);
    const answer=await js(`getActiveSession().messages.at(-1)`);
    assert.doesNotMatch(await js('document.body.innerText'),/\p{Script=Han}/u);
    assert.match(answer.content,/Round 3 decision/);assert.match(answer.content,/H6/);assert.equal(answer.execution.status,'completed');
    assert.ok(await js(`document.querySelectorAll('.msg.assistant .bubble table').length>=4`));
    await until(`Array.from(document.querySelectorAll('.msg.assistant .bubble img')).some(image=>image.complete&&image.naturalWidth>0)`);
    assert.equal(await js(`document.querySelector('.msg.assistant .bubble img').naturalWidth`),1000);
    assert.match(await js(`fetch(document.querySelector('.msg.assistant .bubble img').src).then(r=>r.text())`),/evidence-map-v3/);
    await until(`document.querySelectorAll('.msg.assistant .graph-node').length===6`);
    await js(`document.querySelector('.msg.assistant [data-graph-action="in"]').click();document.querySelector('.msg.assistant .graph-node[data-node="cerebellum"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))`);
    assert.equal(await js(`document.querySelector('.msg.assistant .evidence-graph').dataset.zoom`),'1.25');
    assert.equal(await js(`document.querySelectorAll('.msg.assistant .graph-relation').length`),3);
    assert.ok(await js(`document.querySelectorAll('.msg.assistant .code-block .code-copy').length>0`));
    await until(`document.querySelectorAll('.msg.assistant .output-file-row').length>4&&/KB| B/.test(document.querySelector('.output-file-meta').textContent)`);
    assert.equal(await js(`document.querySelectorAll('.turn-artifacts > .output-file-row').length`),4);
    await js(`document.querySelector('.output-file-open').click()`);
    assert.ok(await js(`!document.querySelector('#output-file-menu').hidden&&document.querySelectorAll('#output-file-menu [role="menuitem"]').length===3`));
    await js(`document.querySelector('#output-file-menu').dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))`);
    checks.push('Grouped output file cards show real file sizes, expandable rows and working open menus');
    assert.ok(await js(`(()=>{const scroller=document.querySelector('.message-table-scroll');scroller.scrollLeft=120;return scroller.scrollLeft>0&&scroller.scrollWidth>scroller.clientWidth;})()`));
    assert.match(await js(`fetch(document.querySelector('.msg.assistant a[href$="IDEA.md"]').href).then(r=>r.text())`),/## H6/);
    assert.match(answer.content,/500 candidates/);assert.match(answer.content,/501 candidates/);
    const counts=await js(`Promise.all(['hypotheses.csv','hypotheses_final.csv'].map(async name=>{const response=await fetch(document.querySelector('.msg.assistant a[href$="/'+name+'"]').href);if(!response.ok)throw Error('Missing candidate artifact');return (await response.text()).trim().split('\\r\\n').length-1;}))`);
    assert.deepEqual(counts,[500,501]);
    checks.push('Full completes with a loaded subgraph, horizontally scrollable tables and H6 artifact');
    checks.push('Eight-row preview links to 500 initial candidates; final export retains 501 candidates');
    await until(`Boolean(document.querySelector('.msg.assistant a[href$="/hypotheses_final.csv"]')?.dataset.artifactKey)`);
    await js(`document.querySelector('.msg.assistant a[href$="/hypotheses_final.csv"]').click()`);
    await until(`Boolean(document.querySelector('.artifact-table tbody tr'))`);
    assert.match(await js(`document.querySelector('.artifact-data-summary').textContent`),/501 rows/);
    assert.ok(await js(`(()=>{const pane=document.querySelector('#artifact-panel').getBoundingClientRect(),chat=document.querySelector('#chat-panel').getBoundingClientRect();return pane.right>=innerWidth-1&&chat.right<=pane.left+1&&document.documentElement.scrollWidth<=innerWidth;})()`));
    await js(`document.querySelector('#artifact-list-tab').click()`);
    assert.ok(await js(`document.querySelectorAll('.artifact-file').length>=8`));
    await js(`[...document.querySelectorAll('.artifact-file')].find(el=>el.querySelector('.artifact-file-name').textContent==='IDEA.md').click()`);
    await until(`Boolean(document.querySelector('.artifact-markdown h1'))`);
    checks.push('Right artifact sidebar opens full 501-row table and formatted Markdown in the shipped executable');
  }
  const report={ok:true,executable,checks,production_frontend:true,model_calls:0,scientific_validation:false};
  fs.writeFileSync(path.join(output,portable?'PORTABLE.json':'PACKAGED.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
  ws.send(JSON.stringify({id:++seq,method:'Page.close'}));
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{if(ws)ws.close();if(child){await delay(1000);if(child.exitCode===null)child.kill();}});
