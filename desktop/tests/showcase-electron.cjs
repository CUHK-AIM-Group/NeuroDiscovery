const {app,session}=require('electron');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const os=require('node:os');
const {createDemoWindow}=require('../demo-main.js');
const output=path.resolve(__dirname,'../../tmp/demo-showcase-check');
app.setPath('userData',fs.mkdtempSync(path.join(os.tmpdir(),'nd-showcase-test-')));
app.disableHardwareAcceleration();
let win;const errors=[],network=[],checks=[];
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function poll(script,timeout=90000){const started=Date.now();while(Date.now()-started<timeout){if(await win.webContents.executeJavaScript(script))return;await sleep(80);}throw new Error('Timeout: '+script);}
async function run(){
  await app.whenReady();fs.mkdirSync(output,{recursive:true});
  session.defaultSession.webRequest.onBeforeRequest({urls:['http://*/*','https://*/*','ws://*/*','wss://*/*']},(d,cb)=>{network.push(d.url);cb({cancel:true});});
  win=createDemoWindow({show:false});win.webContents.setBackgroundThrottling(false);
  win.webContents.on('console-message',details=>{if(details.level==='error')errors.push(details.message);});
  const js=script=>win.webContents.executeJavaScript(script);
  await poll('typeof window.demoStatus === "function"');
  await js(`window.timerRequests=[];window.originalTimer=window.setTimeout;window.setTimeout=(fn,ms,...args)=>{window.timerRequests.push(ms);return window.originalTimer(fn,ms,...args);};void 0;`);
  const capture=async name=>{await js('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');fs.writeFileSync(path.join(output,name+'.png'),(await win.webContents.capturePage()).toPNG());};
  await capture('01-home');
  await js(`document.getElementById('send').click()`);await sleep(150);
  const partial=await js(`document.querySelector('.streaming')?.textContent||''`);
  assert.ok(partial.length>0&&partial.length<120,'text must be streamed progressively');
  await js(`document.getElementById('stop').click()`);const stopped=await js(`document.getElementById('messages').textContent`);await sleep(250);
  assert.equal(await js(`document.getElementById('messages').textContent`),stopped,'stopped playback cannot append');
  assert.equal(await js('demoStatus().completed'),false);checks.push('progressive text and cancellation');
  for(const id of ['chat','idea','data','experiment','full']){
    await js(`document.querySelector('[data-scene="${id}"]').click();document.getElementById('send').click();`);
    const started=Date.now();await poll('demoStatus().completed',120000);
    assert.equal(await js('demoStatus().scenario'),id);
    assert.equal(await js('demoStatus().running'),false);
    if(id==='idea'){
      assert.equal(await js(`document.querySelectorAll('[data-kind=scores] tbody tr').length`),5);
      assert.equal(await js(`document.querySelectorAll('.graph .node').length`),6);
      await js(`document.querySelector('.graph .node').dispatchEvent(new MouseEvent('click',{bubbles:true}))`);
      assert.ok((await js(`document.querySelector('.graph .card-note').textContent`)).includes('ADHD'));
    }
    if(id==='data')assert.equal(await js('document.querySelectorAll(".dataset").length'),4);
    if(id==='experiment')assert.equal(await js('document.querySelectorAll("[data-kind=results] tbody tr").length'),5);
    if(id==='full'){
      assert.equal(await js('document.querySelectorAll(".round").length'),3);
      assert.equal(await js('document.querySelectorAll("[data-kind=scores]")[1].querySelectorAll("tbody tr").length'),3);
    }
    await js(`document.getElementById('scroll-area').scrollTop=document.getElementById('scroll-area').scrollHeight`);await capture(id+'-complete');
    console.log(JSON.stringify({scenario:id,elapsed_ms:Date.now()-started,status:await js('demoStatus()')}));
    checks.push(id+' completed');
  }
  await js(`document.querySelector('.artifact-btn').click()`);assert.equal(await js(`document.getElementById('preview').open`),true);
  assert.ok((await js(`document.getElementById('preview-content').textContent`)).includes('DEMO'));await js(`document.getElementById('close-preview').click()`);
  // The 20-round branch exercises identical logic with an accelerated test clock.
  await js(`window.setTimeout=(fn,ms,...args)=>{window.timerRequests.push(ms);return window.originalTimer(fn,ms*.02,...args);};document.getElementById('full-outcome').value='limit';document.getElementById('full-outcome').dispatchEvent(new Event('change'));document.getElementById('send').click();`);
  await poll('demoStatus().completed',20000);
  assert.equal(await js('document.querySelectorAll(".round").length'),20);
  assert.ok((await js('document.getElementById("messages").textContent')).includes('达到循环上限并停止'));
  checks.push('20-round cap without fabricated success');await capture('full-limit');
  await js(`document.querySelector('[data-scene="idea"]').click();document.getElementById('send').click();`);await sleep(30);
  await js(`document.querySelector('[data-scene="data"]').click()`);await sleep(250);
  assert.equal(await js('document.getElementById("messages").textContent'),'');checks.push('switch cancels stale output');
  await js(`document.getElementById('prompt').value='<img src=x onerror=alert(1)>';document.getElementById('send').click()`);
  assert.equal(await js('demoStatus().running'),false);assert.ok((await js('document.getElementById("status").textContent')).includes('恢复预设'));
  checks.push('unsupported prompt is not misrepresented as answered');
  await js(`document.getElementById('theme').click()`);await capture('dark');
  win.setSize(820,760);await sleep(100);assert.equal(await js('document.body.scrollWidth>window.innerWidth'),false);await capture('compact');
  const timers=await js('window.timerRequests');assert.ok(timers.length>50);assert.ok(Math.max(...timers)<=5000);
  assert.deepEqual(network,[]);assert.deepEqual(errors,[]);
  const report={ok:true,checks,external_requests:network,renderer_errors:errors,max_requested_wait_ms:Math.max(...timers),screenshots:output};
  fs.writeFileSync(path.join(output,'RESULTS.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
}
run().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>{if(win)win.destroy();app.exit(process.exitCode||0);});
