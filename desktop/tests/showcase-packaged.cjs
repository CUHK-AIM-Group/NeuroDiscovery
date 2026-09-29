/* Launch the actual packaged entrypoint, not a browser fixture or Python backend. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const os=require('node:os');
const net=require('node:net');
const {spawn}=require('node:child_process');
const asar=require('@electron/asar');
const desktop=path.resolve(__dirname,'..');
const dist=path.join(desktop,'dist-demo-showcase');
const archive=path.join(dist,'win-unpacked/resources/app.asar');
const output=path.resolve(desktop,'../tmp/demo-showcase-check');
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let child,ws;
const checks=[];
async function main(){
  const files=asar.listPackage(archive).map(p=>p.replaceAll('\\','/').replace(/^\//,''));
  assert.ok(files.includes('demo-main.js'));
  assert.ok(!files.some(f=>/\.env|credentials|core\/web\/server|knowledge_graph/.test(f)));
  for(const file of ['demo-main.js','demo-preload.js',...fs.readdirSync(path.join(desktop,'demo')).map(f=>'demo/'+f)]) {
    assert.ok(asar.extractFile(archive,file).equals(fs.readFileSync(path.join(desktop,file))),'stale packaged file: '+file);
  }
  const pkg=JSON.parse(asar.extractFile(archive,'package.json'));
  assert.equal(pkg.main,'demo-main.js');assert.equal(pkg.distribution,'offline-demo');
  checks.push('packaged source matches checked source; no credentials or backend');
  const server=net.createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));const port=server.address().port;await new Promise(r=>server.close(r));
  const cleanEnv=Object.fromEntries(Object.entries(process.env).filter(([k])=>/^(path|systemroot|windir|comspec|temp|tmp|userprofile|appdata|localappdata|programdata|programfiles|programfiles\(x86\)|commonprogramfiles|systemdrive|number_of_processors|processor_architecture)$/i.test(k)));
  const portable=process.argv.includes('--portable');
  const executable=portable?path.join(dist,'NeuroDiscovery-Offline-Demo-1.0.0-Portable-x64.exe'):path.join(dist,'win-unpacked/NeuroDiscovery Demo.exe');
  const profile=fs.mkdtempSync(path.join(os.tmpdir(),'nd-exe-check-'));
  child=spawn(executable,['--demo-smoke','--remote-debugging-port='+port,'--user-data-dir='+profile],{env:cleanEnv,windowsHide:true,stdio:'ignore'});
  let launchError;child.on('error',e=>{launchError=e;});
  let target;
  for(let i=0;i<100;i++){
    if(launchError)throw launchError;
    try{const targets=await(await fetch('http://127.0.0.1:'+port+'/json')).json();target=targets.find(t=>t.type==='page'&&t.url.includes('index.html'));}catch{}
    if(target)break;await sleep(300);
  }
  assert.ok(target,'packaged window did not load');
  ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
  let seq=0;const pending=new Map();
  ws.addEventListener('message',({data})=>{const message=JSON.parse(data);if(pending.has(message.id)){const p=pending.get(message.id);pending.delete(message.id);clearTimeout(p.timer);message.error?p.reject(new Error(JSON.stringify(message.error))):p.resolve(message.result);}});
  const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(new Error('CDP timeout: '+method));},8000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});
  const js=async expression=>{const response=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(response.exceptionDetails)throw new Error(JSON.stringify(response.exceptionDetails));return response.result.value;};
  for(let i=0;i<30;i++){if(await js('typeof demoStatus === "function"'))break;await sleep(100);}
  assert.deepEqual(await js('demoStatus()'),{scenario:'chat',running:false,completed:false,events:0,artifacts:[]});
  assert.equal(await js('typeof window.demoDesktop.openData'),'function');
  await js('document.getElementById("send").click()');
  await sleep(250);assert.ok((await js('document.querySelector(".streaming")?.textContent||""')).length>0);
  for(let i=0;i<100;i++){if(await js('demoStatus().completed'))break;await sleep(100);}
  assert.equal(await js('demoStatus().completed'),true);checks.push('real executable starts without key and streams chat');
  await js('window.normalTimer=window.setTimeout;window.setTimeout=(fn,ms,...args)=>window.normalTimer(fn,ms*.02,...args);void 0;');
  await js('document.querySelector("[data-scene=full]").click();document.getElementById("send").click();');
  for(let i=0;i<200;i++){if(await js('demoStatus().completed'))break;await sleep(100);}
  assert.equal(await js('demoStatus().completed'),true);
  assert.equal(await js('document.querySelectorAll(".round").length'),3);
  assert.equal(await js('document.querySelectorAll("[data-kind=scores]")[1].querySelectorAll("tbody tr").length'),3);
  await js('document.querySelector(".artifact-btn").click()');
  assert.match(await js('document.getElementById("preview-content").textContent'),/## H6/);
  await js('document.getElementById("close-preview").click();document.getElementById("scroll-area").scrollTop=document.getElementById("scroll-area").scrollHeight;');
  checks.push('packaged Full matches final hypotheses, artifacts, and stop condition');
  const report={ok:true,executable,checks,source_bytes_match:true,model_calls:0,screenshot_scope:'source Electron harness; packaged renderer checked through DOM'};
  fs.writeFileSync(path.join(output,portable?'PORTABLE.json':'PACKAGED.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
  // Closing a page can close the CDP transport before it sends a response.
  ws.send(JSON.stringify({id:++seq,method:'Page.close'}));
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{if(ws)ws.close();if(child){await sleep(1000);if(child.exitCode===null)child.kill();}});
