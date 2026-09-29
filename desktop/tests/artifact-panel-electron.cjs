// Real shared renderer; isolated fixtures, no LLM, scientific data or user profile.
const {app,webContents}=require('electron');
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {setTimeout:delay}=require('node:timers/promises');
const {createDemoServer}=require('../demo/native-server.cjs');
const {createDemoWindow,installNetwork,installIPC}=require('../demo-main.js');
const root=path.resolve(__dirname,'../..'),output=path.join(root,'tmp/artifact-panel-check'),profile=fs.mkdtempSync(path.join(os.tmpdir(),'nd-artifacts-'));
app.setPath('userData',profile);app.disableHardwareAcceleration();
let api,win;const checks=[],errors=[];
const evaluate=source=>win.webContents.executeJavaScript(source);
async function until(source){for(let i=0;i<200;i++){if(await evaluate(source))return;await delay(40);}throw Error('Timeout: '+source);}
async function capture(name){await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');fs.writeFileSync(path.join(output,name+'.png'),(await win.webContents.capturePage()).toPNG());}
async function open(name,selector){
  await evaluate(`document.querySelector('#artifact-toggle').click()`);
  if(await evaluate(`document.querySelector('#artifact-panel').hidden`))await evaluate(`document.querySelector('#artifact-toggle').click()`);
  await evaluate(`document.querySelector('#artifact-list-tab').click();[...document.querySelectorAll('.artifact-file')].find(el=>el.querySelector('.artifact-file-name').textContent===${JSON.stringify(name)}).click()`);
  await until(`Boolean(document.querySelector(${JSON.stringify('#artifact-content '+selector)}))`);
}
function pdf(){
  let value='%PDF-1.4\n',offsets=[0];
  const stream='BT /F1 24 Tf 45 740 Td (NeuroDiscovery - PDF preview) Tj 0 -40 Td /F1 14 Tf (Research report: local UI fixture.) Tj ET';
  const parts=['<< /Type /Catalog /Pages 2 0 R >>','<< /Type /Pages /Kids [3 0 R] /Count 1 >>','<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>','<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',`<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`];
  parts.forEach((part,i)=>{offsets.push(value.length);value+=`${i+1} 0 obj\n${part}\nendobj\n`;});const start=value.length;
  value+='xref\n0 6\n0000000000 65535 f \n'+offsets.slice(1).map(n=>String(n).padStart(10,'0')+' 00000 n \n').join('');
  return value+`trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${start}\n%%EOF\n`;
}
async function run(){
  await app.whenReady();fs.mkdirSync(output,{recursive:true});
  api=await createDemoServer({staticRoot:path.join(root,'core/web/static'),profile,workspace:root,timeScale:.001});
  const assets={
    'REPORT.md':'# Research report\n\nA **readable** report.\n\n| Hypothesis | Result |\n|---|---|\n| H1 | Reviewed |\n\n[Data](scores.csv)\n\n<script>parent.previewPwned=true</script><img src="/preview-leak" onerror="parent.previewPwned=true"><iframe src="/preview-leak"></iframe>',
    'scores.csv':'\uFEFFid,hypothesis,score\r\n'+Array.from({length:503},(_,i)=>`${i+1},"A multiline hypothesis ${i+1}, with a comma\nand a second line",${i}`).join('\r\n'),
    'data.tsv':'id\tvalue\nH1\t12\nH2\t34',
    'config.json':'{"model":"GraphSAGE","training":{"epochs":30},"literal":"<img onerror=alert(1)>"}',
    'model.py':'import torch\n\nclass Model(torch.nn.Module):\n    pass\n',
    'notes.txt':'A plain text research note.\n<literal markup>',
    'plot.svg':'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="250"><rect width="600" height="250" fill="#f0f4fc"/><circle cx="150" cy="120" r="50" fill="#4176e6"/><path d="M200 120h150" stroke="#666" stroke-width="3"/><circle cx="400" cy="120" r="50" fill="#14a38b"/><text x="180" y="220" font-size="20">Research evidence</text><script>parent.previewPwned=true</script></svg>',
    'report.pdf':pdf(),
    'view.html':'<h1>Research result</h1><p>Self-contained HTML preview.</p><script>parent.previewPwned=true;fetch("/preview-leak")</script><img src="/preview-leak"><a href="/preview-leak" target="_top">escape</a>',
    'weights.pt':'binary fallback',
    'large.txt':'x'.repeat(2*1024*1024+1),
    'invalid.json':'{"incomplete":',
  };
  api.runs.set('preview-fixture',{assets});
  installNetwork(api.origin);installIPC({origin:api.origin,profile,getWindow:()=>win});
  win=createDemoWindow({origin:api.origin,show:false});win.webContents.setBackgroundThrottling(false);
  win.webContents.on('console-message',details=>{if(details.level==='error')errors.push(details.message);});
  await until(`typeof state!=='undefined'&&state.settingsLoaded&&Boolean(window.neuroArtifactPanel)`);
  const links=Object.keys(assets).concat('missing.md').map(name=>`[${name}](/demo/artifacts/preview-fixture/${name})`).join('\n\n');
  await evaluate(`startNewChat();getActiveSession().messages.push({role:'assistant',content:${JSON.stringify(links)},id:'artifact-message'});renderActiveSession()`);
  await until(`document.querySelector('.artifact-count').textContent==='13'`);
  const chatId=await evaluate('state.activeSessionId');
  assert.equal(await evaluate('document.querySelector("#artifact-panel").hidden'),true);
  await until(`document.querySelectorAll('.output-file-row').length===13`);
  await until(`document.querySelector('.output-file-meta').textContent.includes(' B')`);
  assert.equal(await evaluate(`document.querySelector('.output-file-meta').textContent`),'Text · MD · '+Buffer.byteLength(assets['REPORT.md'])+' B');
  assert.equal(await evaluate(`document.querySelectorAll('.turn-artifacts > .output-file-row').length`),4);
  assert.equal(await evaluate(`document.querySelectorAll('.output-file-overflow .output-file-row').length`),9);
  assert.ok(await evaluate(`(()=>{const box=document.querySelector('.turn-artifacts'),main=box.closest('.msg-main');return box.getBoundingClientRect().width>=main.getBoundingClientRect().width-4;})()`),'file cards span the conversation column');
  assert.ok(!api.requests.some(r=>String(r).startsWith('GET /demo/artifacts/')),'Card metadata must not download file bodies');
  const fileUrl=api.origin+'/demo/artifacts/preview-fixture/REPORT.md',metadata=await fetch(fileUrl,{method:'HEAD'});
  assert.equal(metadata.status,200);assert.equal(await metadata.text(),'');assert.equal(Number(metadata.headers.get('content-length')),Buffer.byteLength(assets['REPORT.md']));
  await evaluate(`document.querySelector('.turn-artifacts').scrollIntoView({block:'start'})`);await capture('output-file-cards');
  await evaluate(`document.querySelector('.output-file-open').click()`);
  assert.equal(await evaluate(`document.querySelector('#output-file-menu').hidden`),false);
  await capture('output-file-menu');
  await evaluate(`document.querySelector('#output-file-menu').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowDown',bubbles:true}))`);
  assert.equal(await evaluate(`document.activeElement.dataset.fileAction`),'source');
  await evaluate(`document.activeElement.click()`);await until(`Boolean(document.querySelector('#artifact-content .artifact-text'))`);
  assert.equal(await evaluate(`document.querySelector('#artifact-content code').textContent`),assets['REPORT.md']);
  await evaluate(`document.querySelector('#artifact-close').click();document.querySelector('.output-file').click()`);
  await until(`Boolean(document.querySelector('.artifact-markdown h1'))`);
  await evaluate(`document.querySelector('.output-file-open').click();document.querySelector('#output-file-menu').dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))`);
  assert.equal(await evaluate(`document.querySelector('#artifact-panel').hidden`),false,'Escape dismisses only the file menu');
  assert.equal(await evaluate(`document.activeElement.className`),'output-file-open');
  await evaluate(`document.querySelector('#artifact-close').click();document.querySelector('.output-file-overflow').open=true;document.querySelector('.output-file-open').click();renderMarkdownBubble(document.querySelector('.msg.assistant .bubble'),${JSON.stringify(links+'\n\nStreaming update')})`);
  await until(`Boolean(document.querySelector('.turn-artifacts'))`);
  assert.equal(await evaluate(`document.querySelector('.output-file-overflow').open`),true);
  assert.equal(await evaluate(`document.querySelector('#output-file-menu').hidden`),false);
  await evaluate(`document.querySelector('#output-file-menu').dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));document.querySelector('.output-file-overflow').open=false`);
  const downloadPath=path.join(output,'card-report.md');
  const cardDownload=new Promise((resolve,reject)=>win.webContents.session.once('will-download',(_event,item)=>{item.setSavePath(downloadPath);item.once('done',(_event,state)=>state==='completed'?resolve():reject(Error('Download '+state)));}));
  await evaluate(`document.querySelector('.output-file-open').click();document.querySelector('[data-file-action="download"]').click()`);await cardDownload;
  assert.equal(fs.readFileSync(downloadPath,'utf8'),assets['REPORT.md']);
  checks.push('Camellia-style reply file cards: real HEAD sizes, four-row collapse, keyboard menu, preview/source/download, streaming state');
  await open('scores.csv','.artifact-table');
  assert.match(await evaluate(`document.querySelector('.artifact-data-summary').textContent`),/503 rows/);
  assert.equal(await evaluate(`document.querySelectorAll('.artifact-table tbody tr').length`),100);
  await evaluate(`document.querySelector('.artifact-pagination button:last-child').click()`);
  assert.equal(await evaluate(`document.querySelector('.artifact-table tbody td').textContent`),'101');
  await evaluate(`const search=document.querySelector('#artifact-table-search');search.value='hypothesis 503,';search.dispatchEvent(new Event('input'));`);
  assert.equal(await evaluate(`document.querySelectorAll('.artifact-table tbody tr').length`),1);
  assert.match(await evaluate(`document.querySelector('.artifact-table tbody').textContent`),/503/);
  assert.ok(await evaluate(`(()=>{const s=document.querySelector('.artifact-table-scroll');s.scrollLeft=400;return s.scrollLeft>0;})()`));
  checks.push('CSV: complete 503-row search, pagination, multiline fields, horizontal scroll');
  await capture('csv-search');
  await open('REPORT.md','.artifact-markdown');
  assert.equal(await evaluate(`document.querySelector('.artifact-markdown h1').textContent`),'Research report');
  assert.equal(await evaluate(`document.querySelectorAll('.artifact-markdown script,.artifact-markdown iframe,.artifact-markdown [onerror]').length`),0);
  await evaluate(`document.querySelector('.artifact-markdown a').click()`);await until(`Boolean(document.querySelector('.artifact-table'))`);
  checks.push('Markdown: formatted report, nested artifact links, no active HTML');
  for(const [name,selector] of [['data.tsv','.artifact-table'],['config.json','.artifact-text'],['model.py','.artifact-text'],['notes.txt','.artifact-text'],['invalid.json','.artifact-notice']]){
    await open(name,selector);checks.push(name+' preview');
  }
  await open('plot.svg','.artifact-image-stage img');await until(`document.querySelector('.artifact-image-stage img').naturalWidth===600`);
  await evaluate(`document.querySelector('.artifact-image-tools button:last-child').click()`);
  assert.ok(await evaluate(`document.querySelector('.artifact-image-stage').classList.contains('is-zoomed')`));
  await capture('image');checks.push('SVG image preview and actual-size zoom');
  await open('view.html','.artifact-frame');
  assert.equal(await evaluate(`document.querySelector('.artifact-frame').getAttribute('sandbox')`),'');
  await delay(300);assert.equal(await evaluate('Boolean(window.previewPwned)'),false);
  assert.ok(!api.requests.some(r=>String(r).includes('preview-leak')));
  await capture('html');checks.push('HTML: opaque-origin sandbox, scripts/navigation/network blocked');
  await open('report.pdf','.artifact-frame');await delay(1500);await capture('pdf');
  const pdfFrames=[];function frames(frame){for(const child of frame.frames){pdfFrames.push(child);frames(child);}}frames(win.webContents.mainFrame);
  const pdfFrame=pdfFrames.find(f=>f.url.startsWith('blob:'));
  assert.ok(pdfFrame,'PDF iframe loaded');
  const plugin=await pdfFrame.executeJavaScript(`Boolean(document.querySelector('embed[type="application/pdf"]'))`);
  const guests=[];
  for(const wc of webContents.getAllWebContents())if(wc.getURL().startsWith('chrome-extension://mhjfbmdgcfjbbpaeojofohoefgiehjai/'))guests.push(await wc.executeJavaScript(`(()=>{const v=document.querySelector('pdf-viewer');return {loaded:v?.initialLoadComplete_,pages:v?.documentDimensions?.pageDimensions?.length}})()`));
  assert.equal(plugin,true,'Chromium PDF viewer mounted');assert.ok(guests.some(v=>v.loaded&&v.pages===1),JSON.stringify(guests));checks.push('PDF: native viewer loaded the one-page document');
  await open('large.txt','.artifact-empty');await until(`document.querySelector('#artifact-content').textContent.includes('too large')`);
  await open('weights.pt','.artifact-empty');assert.match(await evaluate(`document.querySelector('#artifact-content').textContent`),/Download/);
  await open('missing.md','.artifact-empty');await until(`document.querySelector('#artifact-content').textContent.includes('unavailable')`);
  checks.push('Unavailable, binary and oversized files retain download/retry controls');
  await open('scores.csv','.artifact-table');
  for(const width of [1600,1320,960]){
    win.setSize(width,900);await delay(100);
    assert.ok(await evaluate(`document.documentElement.scrollWidth<=document.documentElement.clientWidth+1`),'no viewport overflow at '+width);
    assert.ok(await evaluate(`document.querySelector('#artifact-close').getBoundingClientRect().right<=innerWidth`));
    assert.ok(await evaluate(`document.querySelector('#artifact-panel').getBoundingClientRect().right>=innerWidth-1`),'sidebar stays on the right');
    if(width>1200)assert.ok(await evaluate(`document.querySelector('#chat-panel').getBoundingClientRect().right<=document.querySelector('#artifact-panel').getBoundingClientRect().left+1`),'docked preview does not obscure chat');
    await capture('sidebar-'+width);
  }
  win.setSize(1600,900);await delay(100);
  const before=await evaluate(`document.querySelector('#artifact-panel').getBoundingClientRect().width`);
  await evaluate(`document.querySelector('.artifact-resizer').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true}))`);
  assert.equal(await evaluate(`document.querySelector('#artifact-panel').getBoundingClientRect().width`),before+20);
  await evaluate(`applyTheme('dark');document.querySelector('#artifact-list-tab').click()`);await capture('artifacts-dark');
  await evaluate(`applyTheme('light');document.querySelector('#artifact-list-tab').click()`);await capture('artifacts-light');
  await evaluate(`setActiveView('settings')`);await until(`document.querySelector('#artifact-panel').hidden&&document.querySelector('#artifact-toggle').hidden`);
  await evaluate(`setActiveView('chat')`);await until(`!document.querySelector('#artifact-panel').hidden`);
  await evaluate(`document.querySelector('#artifact-close').dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))`);assert.equal(await evaluate(`document.querySelector('#artifact-panel').hidden`),true);
  const center=await evaluate(`getComputedStyle(document.documentElement).getPropertyValue('--chat-read-w').trim()`);assert.equal(center,'1120px');
  await evaluate(`startNewChat()`);await until(`document.querySelector('.artifact-count').textContent==='0'`);assert.equal(await evaluate(`document.querySelector('#artifact-panel').hidden`),true);
  await evaluate(`state.activeSessionId=${JSON.stringify(chatId)};renderActiveSession()`);await until(`document.querySelector('.artifact-count').textContent==='13'`);
  checks.push('Dock/drawer widths, keyboard resize, Escape, themes, view changes, per-chat artifact isolation and widened chat');
  assert.equal(await evaluate('Boolean(window.previewPwned)'),false);
  assert.ok(!api.requests.some(r=>String(r).includes('preview-leak')));
  assert.ok(!api.requests.some(r=>String(r).startsWith('POST /api/chat')));
  const unexpected=errors.filter(e=>!/(Content Security Policy|security policy|Blocked script execution in 'about:srcdoc'|404|ERR_BLOCKED_BY_CLIENT)/i.test(e));assert.deepEqual(unexpected,[]);
  fs.writeFileSync(path.join(output,'RESULTS.json'),JSON.stringify({ok:true,checks,errors,requests:api.requests},null,2));console.log(JSON.stringify({ok:true,checks}));
}
run().then(async()=>{win.destroy();await api.close();app.exit(0);}).catch(async error=>{console.error(error);if(win){await capture('failure').catch(()=>{});win.destroy();}if(api)await api.close();app.exit(1);});
