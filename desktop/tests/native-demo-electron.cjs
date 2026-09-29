const {app}=require('electron');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const os=require('node:os');
const {setTimeout:delay}=require('node:timers/promises');
const {createDemoServer,presets,limitPrompt}=require('../demo/native-server.cjs');
const D=require('../demo/scenarios.js');
const {createDemoWindow,installNetwork,installIPC}=require('../demo-main.js');
const root=path.resolve(__dirname,'../..'),output=path.join(root,'tmp/demo-native-check');
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'nd-native-ui-'));
app.setPath('userData',profile);app.disableHardwareAcceleration();
app.on('window-all-closed',()=>{}); // Keep the test process alive while replacing the window after restart.
let win,api,ideaSessionId;const errors=[],checks=[];
const evaluate=source=>win.webContents.executeJavaScript(source);
async function until(source,timeout=45000){
  const start=Date.now();while(!(await evaluate(source))){if(Date.now()-start>timeout)throw Error('Timed out: '+source);await delay(40);}
}
async function screenshot(name){await evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');fs.writeFileSync(path.join(output,name+'.png'),(await win.webContents.capturePage()).toPNG());}
async function choose(mode){
  await evaluate(`startNewChat();document.querySelector('#model-menu-btn').click();document.querySelector('[data-model-panel="autoresearch"]').click();document.querySelector('[data-autoresearch-mode="${mode}"]').click();`);
  assert.equal(await evaluate('state.autoResearchMode'),mode);
}
async function send(prompt){
  await evaluate(`document.querySelector('#msg-input').value=${JSON.stringify(prompt)};document.querySelector('#msg-input').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('#send-btn').click();`);
}
async function result(){
  await until(`!isSessionWaiting() && getActiveSession().messages.some(m=>m.role==='assistant')`);
  return evaluate(`getActiveSession().messages.findLast(m=>m.role==='assistant')`);
}
async function run(){
  fs.mkdirSync(output,{recursive:true});await app.whenReady();
  api=await createDemoServer({staticRoot:path.join(root,'core/web/static'),profile,workspace:root,timeScale:run=>['chat','idea'].includes(run.scenario)?1:.025});
  installNetwork(api.origin);installIPC({origin:api.origin,profile,getWindow:()=>win});
  win=createDemoWindow({origin:api.origin,show:false});win.webContents.setBackgroundThrottling(false);
  win.webContents.on('console-message',details=>{if(details.level==='error')errors.push(details.message);});
  await until(`typeof state!=='undefined'&&state.settingsLoaded&&state.activeModel==='GPT-6-Astra'&&Boolean(document.querySelector('.harness-welcome'))`);
  assert.equal(await evaluate('Boolean(window.marked?.parse)'),true);
  assert.equal(await evaluate('state.llmConnectionStatus.apiKeyRequired'),false);
  assert.equal(await evaluate('document.documentElement.lang'),'en');
  assert.equal(await evaluate('Boolean(document.querySelector(".scene-button, #scenarios, #demo-banner"))'),false);
  assert.equal(await evaluate('document.querySelector(".harness-welcome h1").textContent'),'NeuroDiscovery');
  await delay(250);await screenshot('native-home');
  // Hypothesis selection is offered only where hypotheses are produced.
  for(const mode of ['off','data','model']){
    await choose(mode);
    await evaluate(`document.querySelector('#model-menu-btn').click()`);
    assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('#model-menu [data-model-panel]'),x=>x.dataset.modelPanel)`),['models','autoresearch']);
  }
  await choose('end-to-end');
  await evaluate(`document.querySelector('#model-menu-btn').click()`);
  assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('#model-menu [data-model-panel]'),x=>x.dataset.modelPanel)`),['models','autoresearch','generation']);
  await evaluate(`document.querySelector('#model-menu-btn').click()`);
  await choose('idea');
  await evaluate(`document.querySelector('#model-menu-btn').click()`);
  assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('#model-menu [data-model-panel]'),x=>x.dataset.modelPanel)`),['models','autoresearch','generation']);
  assert.equal(await evaluate(`document.querySelector('[data-model-panel="generation"] .model-menu-name').textContent`),'Hypothesis selection');
  assert.equal(await evaluate(`getActiveSession().reasoningEffort`),'high');
  await screenshot('native-model-menu');
  await evaluate(`document.querySelector('[data-model-panel="generation"]').click()`);
  assert.equal(await evaluate(`document.querySelector('#model-menu .model-menu-title').textContent`),'Hypothesis selection');
  await screenshot('native-hypothesis-selection');
  await evaluate(`document.querySelector('#model-menu-btn').click()`);
  for(const p of presets){
    await choose(p.mode);await send(p.prompt);
    if(p.id==='chat'){
      await until(`Boolean(document.querySelector('#wb-live-execution .bubble')?.textContent)`);
      const before=await evaluate(`document.querySelector('#wb-live-execution .bubble').textContent.length`);
      await until(`(document.querySelector('#wb-live-execution .bubble')?.textContent.length||0)>${before}`);
      await screenshot('native-streaming');
    }
    if(p.id==='idea'){
      await until(`Boolean(document.querySelector('#wb-live-execution .message-table-prose'))`);
      const before=await evaluate(`(()=>{const bubble=document.querySelector('#wb-live-execution .bubble'),table=bubble.querySelector('.message-table-scroll');table.scrollLeft=120;return {length:bubble.dataset.text.length,left:table.scrollLeft};})()`);
      assert.ok(before.left>0,'streaming table must scroll');
      await until(`(document.querySelector('#wb-live-execution .bubble')?.dataset.text.length||0)>${before.length}`);
      assert.equal(await evaluate(`document.querySelector('#wb-live-execution .message-table-scroll').scrollLeft`),before.left,'stream updates preserve horizontal scroll');
      checks.push({streaming_table_scroll:true});
    }
    const answer=await result();
    const expectedTitle=D.definitions.find(definition=>definition.id===p.id).titles.en;
    await until(`getActiveSession().autoTitlePending===false&&getActiveSession().title===${JSON.stringify(expectedTitle)}`);
    assert.ok(await evaluate(`Array.from(document.querySelectorAll('.chat-title')).some(node=>node.textContent===${JSON.stringify(expectedTitle)})`));
    assert.doesNotMatch(JSON.stringify({content:answer.content,blocks:answer.execution.blocks,tools:answer.execution.tools}),/演示|模拟|预设|合成|回放|玩具|API\s*key|synthetic|executed|demo_model/i);
    assert.equal(answer.execution.status,'completed');
    assert.doesNotMatch(await evaluate('document.body.innerText'),/\p{Script=Han}/u,`${p.id} visible UI must be English`);
    if(p.id==='chat')await screenshot('native-chat-recording');
    if(p.id==='idea'){
      assert.equal(answer.execution.tools.filter(t=>t.tool.startsWith('Agent ')).length,3);
      assert.match(answer.content,/500 candidates/);assert.match(answer.content,/492 candidates/);
      assert.equal(await evaluate(`document.querySelectorAll('.msg.assistant .bubble table tbody tr').length`),8);
      const csvPath=path.join(output,'native-hypotheses.csv');
      const downloaded=new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>reject(Error('Hypothesis CSV download timed out')),10000);
        win.webContents.session.once('will-download',(_event,item)=>{
          item.setSavePath(csvPath);
          item.once('done',(_event,status)=>{clearTimeout(timer);status==='completed'?resolve():reject(Error('Download '+status));});
        });
      });
      await until(`Boolean(document.querySelector('.msg.assistant a[href$="/hypotheses.csv"]')?.dataset.artifactKey)`);
      await evaluate(`document.querySelector('.msg.assistant a[href$="/hypotheses.csv"]').click()`);
      await until(`Boolean(document.querySelector('#artifact-content .artifact-table tbody tr'))`);
      assert.equal(await evaluate(`document.querySelectorAll('#artifact-content .artifact-table tbody tr').length`),100);
      assert.match(await evaluate(`document.querySelector('.artifact-data-summary').textContent`),/500 rows/);
      await screenshot('native-artifact-hypotheses');
      await evaluate(`document.querySelector('#artifact-download').click()`);await downloaded;
      await evaluate(`document.querySelector('#artifact-close').click()`);
      assert.equal(fs.readFileSync(csvPath,'utf8'),D.hypothesisCsv(D.hypothesisPool));
      checks.push({hypothesis_pool:500,preview_rows:8,complete_csv_downloaded:true});
      checks.push({artifact_sidebar:true,table_page_rows:100,complete_preview_rows:500});
      assert.ok(await evaluate(`document.querySelectorAll('.msg.assistant .bubble table').length>=1`));
      await until(`Array.from(document.querySelectorAll('.msg.assistant .bubble img')).some(x=>x.complete&&x.naturalWidth>0)`);
      ideaSessionId=await evaluate('state.activeSessionId');
      await until(`Boolean(document.querySelector('.msg.assistant .evidence-graph'))`);
      await evaluate(`document.querySelector('.msg.assistant .evidence-graph').scrollIntoView({block:'center'})`);
      await screenshot('native-idea');
      for(const width of [1320,960]){
        win.setSize(width,900);await delay(100);
        const layout=await evaluate(`(()=>{const bubble=document.querySelector('.msg.assistant .bubble'),scroll=bubble.querySelector('.message-table-scroll');scroll.scrollIntoView({block:'center'});scroll.scrollLeft=10000;const singleLine=cell=>{const range=document.createRange();range.selectNodeContents(cell);return range.getClientRects().length===1;};return {viewport:scroll.clientWidth,content:scroll.scrollWidth,left:scroll.scrollLeft,bubbleOverflow:bubble.scrollWidth-bubble.clientWidth,pageOverflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,headersSingleLine:[...scroll.querySelectorAll('th')].every(singleLine),scoresSingleLine:[...scroll.querySelectorAll('td')].filter(td=>/^\\d+(\\.\\d+)?$/.test(td.textContent.trim())).every(singleLine),proseWidth:scroll.querySelector('.message-table-prose').getBoundingClientRect().width};})()`);
        assert.ok(layout.content>layout.viewport&&layout.left>0,JSON.stringify(layout));
        assert.ok(layout.bubbleOverflow<=1&&layout.pageOverflow<=1,JSON.stringify(layout));
        assert.equal(layout.headersSingleLine,true);assert.equal(layout.scoresSingleLine,true);assert.ok(layout.proseWidth>=300);
        await screenshot('native-table-right-'+width);
        await evaluate(`document.querySelector('.message-table-scroll').scrollLeft=0`);await screenshot('native-table-left-'+width);
        checks.push({table_viewport:width,...layout});
      }
      win.setSize(1320,900);
    }
    if(p.id==='data')assert.match(answer.content,/model_ready/);
    if(p.id==='experiment')assert.match(answer.content,/Final test AUROC/);
    if(p.id==='full'){
      assert.match(answer.content,/Round 3 decision/);assert.match(answer.content,/501 candidates/);
      await evaluate(`[...document.querySelectorAll('.msg.assistant strong')].find(node=>node.textContent==='Round 2 decision').scrollIntoView({block:'start'})`);
      await screenshot('native-round-feedback');
      await evaluate(`document.querySelector('.msg.assistant a[href$="/round_02.csv"]').click()`);
      await until(`Boolean(document.querySelector('.artifact-table tbody tr'))`);
      assert.equal(await evaluate(`document.querySelectorAll('.artifact-table tbody tr').length`),3);
      assert.match(await evaluate(`document.querySelector('#artifact-content').textContent`),/0.049/);
      await screenshot('native-round-results');
      await evaluate(`document.querySelector('#artifact-close').click();document.querySelector('.msg.assistant a[href$="/REPORT.md"]').click()`);
      await until(`Boolean(document.querySelector('.artifact-markdown h1'))`);
      assert.match(await evaluate(`document.querySelector('.artifact-markdown').textContent`),/Round 3 decision/);
      await evaluate(`document.querySelector('#artifact-close').click()`);
      checks.push({round_feedback:true,round_result_rows:3,full_report:true});
      const counts=await evaluate(`Promise.all(['hypotheses.csv','hypotheses_final.csv'].map(async name=>{const link=document.querySelector('.msg.assistant a[href$="/'+name+'"]');const response=await fetch(link.href);if(!response.ok)throw Error('Missing full artifact');return (await response.text()).trim().split('\\r\\n').length-1;}))`);
      assert.deepEqual(counts,[500,501]);checks.push({initial_pool_rows:counts[0],final_pool_rows:counts[1]});
    }
    checks.push({scenario:p.id,mode:p.mode,title:expectedTitle,english_only:true,tools:answer.execution.tools.length,tables:await evaluate(`document.querySelectorAll('.msg.assistant .bubble table').length`)});
    console.log('PASS native UI '+p.id);
  }
  await choose('end-to-end');await send(limitPrompt);
  assert.match((await result()).content,/Round 20 decision/);checks.push({branch:'20-round-limit',passed:true});
  await choose('idea');await send(presets[1].prompt);
  await until(`Boolean(sessionRequestFor()?.execution?.tools?.length)`);
  await evaluate(`document.querySelector('#send-btn').click()`);
  assert.equal((await result()).execution.status,'cancelled');checks.push({stop:true});
  await evaluate(`workbenchHistory.flush()`);
  const count=await evaluate('state.sessions.length');
  await win.loadURL(api.origin+'/harness');await until(`state.settingsLoaded&&state.sessions.length>=${count}`);
  assert.equal(await evaluate(`state.sessions.find(session=>session.id===${JSON.stringify(ideaSessionId)}).title`),D.definitions[1].titles.en);
  checks.push({persisted_sessions:count});
  await evaluate(`state.activeSessionId=${JSON.stringify(ideaSessionId)};renderActiveSession();saveSessions();workbenchHistory.flush()`);
  const oldRequests=api.requests.slice();
  win.destroy();await api.close();
  api=await createDemoServer({staticRoot:path.join(root,'core/web/static'),profile,workspace:root});
  installNetwork(api.origin);win=createDemoWindow({origin:api.origin,show:false});win.webContents.setBackgroundThrottling(false);
  await until(`typeof state!=='undefined'&&state.settingsLoaded&&state.activeSessionId===${JSON.stringify(ideaSessionId)}`);
  await until(`Array.from(document.querySelectorAll('.msg.assistant .bubble img')).some(x=>x.complete&&x.naturalWidth>0)`);
  assert.equal(api.runs.size,0);
  assert.ok(await evaluate(`(()=>{const scroll=document.querySelector('.message-table-scroll');scroll.scrollLeft=100;return scroll.scrollLeft>0&&scroll.querySelectorAll('.message-table-scroll').length===0;})()`));
  await until(`Boolean(document.querySelector('.msg.assistant .evidence-graph'))`);
  await evaluate(`document.querySelector('.msg.assistant .evidence-graph').scrollIntoView({block:'center'})`);await screenshot('native-idea-after-restart');
  checks.push({backend_restart_graph:true,restored_table_scroll:true});
  assert.deepEqual(errors.filter(x=>!x.includes('404')),[]);
  fs.writeFileSync(path.join(output,'RESULTS.json'),JSON.stringify({ok:true,production_frontend:true,keyless:true,checks,errors,requests:[...new Set([...oldRequests,...api.requests])]},null,2));
  console.log(JSON.stringify({ok:true,checks}));
}
run().then(async()=>{win.destroy();await api.close();app.exit(0);}).catch(async error=>{
  console.error(error);if(win){console.error(await evaluate(`({view:state.activeView,messages:getActiveSession()?.messages,status:document.querySelector('#status-text')?.textContent})`).catch(()=>null));await screenshot('failure').catch(()=>{});win.destroy();}
  if(api)await api.close();app.exit(1);
});
