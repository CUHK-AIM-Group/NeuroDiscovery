(() => {
  'use strict';
  const D=window.DemoScenarios, P=window.DemoPlayer;
  const $=id=>document.getElementById(id);
  const state={selected:'chat', controller:null, completed:false, events:[], files:{}, preview:null};
  const scroll=$('scroll-area');
  let follow=true;
  scroll.addEventListener('scroll',()=>{follow=scroll.scrollHeight-scroll.scrollTop-scroll.clientHeight<100;});
  const stick=()=>{if(follow) scroll.scrollTop=scroll.scrollHeight;};
  const node=(tag,cls,text)=>{const el=document.createElement(tag);if(cls)el.className=cls;if(text!==undefined)el.textContent=text;return el;};
  const append=el=>{$('messages').append(el);stick();return el;};
  function card(title,note='模拟结果') {
    const el=node('section','visual-card');
    const heading=node('div','card-title',title);heading.append(node('small','',note));el.append(heading);return el;
  }
  function table(headers,rows,{classes=[],title='结果表',note='模拟结果'}={}) {
    const el=card(title,note), area=node('div','table-scroll'), t=node('table');
    const thead=node('thead'), tr=node('tr');headers.forEach(h=>{const th=node('th','',h);th.scope='col';tr.append(th);});thead.append(tr);t.append(thead);
    const tbody=node('tbody');rows.forEach(row=>{const tr=node('tr');row.forEach((v,i)=>tr.append(node('td',classes[i]||'',String(v))));tbody.append(tr);});
    t.append(tbody);area.append(t);el.append(area);return el;
  }
  const notes={
    chat:['普通聊天，不启动研究流程','NeuroDiscovery · NeuroOracle · NeuroRuntime',['角色介绍','能力说明','选择工作模式']],
    idea:['ADHD · 小脑—默认模式网络','图谱 → GNN → 新颖性 → 三 agent',['证据与边界','多维评分','3 个候选假设']],
    data:['4 个队列 · MRI / fMRI / EEG','模态检查 → 处理 → 质控 → 划分',['下载状态','处理过程','模型输入结构']],
    experiment:['IDEA.md + ADHD-200 样例','模型选择 → 代码 → 调参 → 评估',['模型代码','训练过程','实验与验证表']],
    full:['目标 3 个 · 最多 20 轮','Idea → Data → Experiment → 验证',['完整工作流','逐轮决策','达标或到上限']],
  };
  function busy(value) {
    document.body.classList.toggle('busy',value);
    $('stop').hidden=!value;$('send').hidden=value;
    for(const id of ['prompt','mode','model','reset-prompt','full-outcome']) $(id).disabled=value;
  }
  function stop() {
    if(!state.controller)return;
    state.controller.abort();state.controller=null;busy(false);
    document.querySelectorAll('.streaming').forEach(el=>el.classList.remove('streaming'));
    document.querySelectorAll('.tool.running').forEach(el=>{el.classList.replace('running','stopped');el.querySelector('.tool-icon').textContent='■';el.querySelector('small').textContent='已停止';});
    $('status').textContent='已停止 · 再次发送可从头重播';
    state.completed=false;$('export').disabled=true;
  }
  function select(id) {
    stop();const def=D.definitions.find(s=>s.id===id);if(!def)return;
    state.selected=id;state.completed=false;state.events=[];state.files={};
    $('messages').replaceChildren();$('welcome').hidden=false;$('mode').value=id;$('prompt').value=def.prompt;
    $('scene-title').textContent=def.name;$('scene-number').textContent=String(D.definitions.indexOf(def)+1).padStart(2,'0')+' / 05';
    $('scene-description').textContent=def.subtitle;$('scene-overview').replaceChildren();
    notes[id][2].forEach((title,i)=>{const el=node('div','overview-card');el.append(node('strong','',title),node('span','',['观察工作过程','查看中间产物','理解最终决策'][i]));$('scene-overview').append(el);});
    $('context').replaceChildren();
    for(const [label,value] of [['模型','GPT-6-Astra · 演示'],['模式',def.mode==='off'?'AutoResearch 已关闭':def.name],['研究范围',notes[id][0]],['流程',notes[id][1]]]) {
      const el=node('div','context-row');el.append(node('small','',label),node('div','',value));$('context').append(el);
    }
    $('full-control').hidden=id!=='full';$('artifacts').replaceChildren(node('p','muted','播放完成后可查看并导出。'));
    $('artifact-count').textContent='0';$('export').disabled=true;$('status').textContent='预设问题已填入 · 点击 ↑ 开始';$('progress-label').textContent='';$('progress').style.width='0%';
    document.querySelectorAll('.scene-button').forEach(el=>{if(el.dataset.scene===id)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
    follow=true;scroll.scrollTop=0;
  }
  function showGraph() {
    const el=card('NeuroOracle · 证据子图','示意节点 / 关系，可点击');el.classList.add('graph');
    const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 665 355');svg.setAttribute('role','img');svg.setAttribute('aria-label','ADHD 小脑 默认模式网络 注意症状示意图');
    const make=(tag,attrs,text)=>{const x=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>x.setAttribute(k,v));if(text)x.textContent=text;return x;};
    const byId=Object.fromEntries(D.graph.nodes.map(n=>[n[0],n]));
    D.graph.edges.forEach(([from,to,label])=>{const a=byId[from],b=byId[to];svg.append(make('line',{x1:a[2],y1:a[3],x2:b[2],y2:b[3]}));svg.append(make('text',{x:(a[2]+b[2])/2,y:(a[3]+b[3])/2-9,class:'edge-label','text-anchor':'middle'},label));});
    const detail=node('p','card-note','点击节点查看关联路径。检索未发现某关系，不等于证明它具有新颖性。');
    D.graph.nodes.forEach(([id,label,x,y])=>{const g=make('g',{class:'node',tabindex:0,role:'button','aria-label':label});g.append(make('circle',{cx:x,cy:y,r:25}),make('text',{x,y:y+46},label),make('text',{x,y:y+5},label.slice(0,1)));const activate=()=>{detail.textContent=label+' · '+D.graph.edges.filter(e=>e[0]===id||e[1]===id).map(e=>byId[e[0]][1]+' → '+byId[e[1]][1]+'（'+e[2]+'）').join('；')+'。以上为演示证据线索，未加载真实图谱。';};g.addEventListener('click',activate);g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}});svg.append(g);});
    el.append(svg,detail);append(el);
  }
  async function render(event,sleep) {
    if(event.kind==='message') {
      $('status').textContent='正在组织并输出回复…';
      const el=append(node('div','message streaming'));el.dataset.kind='message';
      const chars=Array.from(event.text);
      for(let i=0;i<chars.length;i+=7){await sleep(32);el.textContent+=chars.slice(i,i+7).join('');stick();}
      el.classList.remove('streaming');
    } else if(event.kind==='tool') {
      $('status').textContent='正在演示 '+event.name+'…';
      const el=node('details','tool running'),summary=node('summary'),icon=node('span','tool-icon');
      summary.append(icon,node('span','',event.name),node('small','','模拟执行中'));
      const pre=node('pre','',JSON.stringify(event.input,null,2));el.append(summary,pre);el.dataset.kind='tool';append(el);
      await sleep(event.delay);el.classList.remove('running');icon.textContent='✓';summary.querySelector('small').textContent=(event.delay/1000).toFixed(1)+'s · 模拟';pre.textContent+='\n\n'+event.output;stick();
    } else if(event.kind==='graph') {showGraph();await sleep(450);
    } else if(event.kind==='scores') {
      const el=table(['序号 / ID','Hypothesis','新颖性','结构','GNN','统计','临床','方法学','评审均分','总分','筛选'],(event.rows||D.hypotheses).map((h,i)=>[`${i+1} / ${h.id}`,h.text,h.novelty,h.structural,h.gnn,h.statistical,h.clinical,h.methodological,h.review.toFixed(2),h.total.toFixed(2),h.status]),{title:event.title||'Hypothesis 排名',note:'按总分降序 · 0–100',classes:['numeric','hypothesis',...Array(7).fill('numeric'),'numeric selected-score','']});el.dataset.kind='scores';el.append(node('p','card-note',D.scoreNote));append(el);await sleep(600);
    } else if(event.kind==='datasets') {
      const list=append(node('div','dataset-list'));list.dataset.kind='datasets';
      for(const d of D.datasets) {const el=node('article','dataset'),heading=node('h3','',d.name);heading.append(node('span','',d.modalities));el.append(heading);
        for(const [label,value] of [['真实下载状态',d.downloaded],['需要处理',d.pipeline],['随包样例',d.fixture],['模型输入',d.ready+' → '+d.models]]){const p=node('p');p.append(node('b','',label+'：'),document.createTextNode(value));el.append(p);}
        const link=node('a','','官方数据说明 ↗');link.href=d.source;link.target='_blank';link.rel='noreferrer noopener';el.append(link);list.append(el);stick();await sleep(400);}
    } else if(event.kind==='tree'||event.kind==='code') {
      const el=card(event.filename||'最终数据组织结构',event.kind==='tree'?'随包合成样例':'示例代码 / 不在播放器内执行');el.classList.add(event.kind);el.dataset.kind=event.kind;el.append(node('pre','',event.text));append(el);await sleep(500);
    } else if(event.kind==='capabilities') {
      append(table(['模式','工作范围','主要交付'],[['Idea','图谱证据、候选与评审','IDEA.md / hypothesis 排名'],['Data','多模态整理、处理与质控','model_ready / 划分与 manifest'],['Experiment','模型实现、调参、评估','代码 / 配置 / 结果表'],['Full','串联以上环节并按验证反馈循环','研究报告 / 每轮记录']],{title:'按研究阶段选择模式',note:'本 Demo 覆盖全部五个场景'}));await sleep(350);
    } else if(event.kind==='training') {
      $('status').textContent='正在回放训练与验证集调参…';const el=card('训练与调参轨迹','模拟验证 AUROC · 不是实测曲线');el.dataset.kind='training';
      const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 660 215');svg.classList.add('training-chart');
      const s=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n;};
      for(const val of [.5,.6,.7,.8]){const y=175-(val-.5)*400;svg.append(s('line',{x1:45,x2:630,y1:y,y2:y,class:'axis'}),s('text',{x:12,y:y+3},val.toFixed(1)));}
      const points=[],line=s('polyline',{points:'',class:'curve'});svg.append(line);el.append(svg);const label=node('div','training-label');el.append(label);append(el);
      const values=[.53,.621,.693,.754,.786,.809];for(let i=0;i<values.length;i++){await sleep(500);points.push((55+i*110)+','+(175-(values[i]-.5)*400));line.setAttribute('points',points.join(' '));label.textContent=`Epoch ${String((i+1)*5).padStart(2,'0')} / 30 · validation AUROC ${values[i].toFixed(3)} · test 未查看`;stick();}
    } else if(event.kind==='results') {
      const el=table(D.resultHeaders,D.results,{title:'实验结果对比',note:'全表为模拟值，未运行训练',classes:['','numeric','numeric','numeric','numeric']});el.dataset.kind='results';el.append(node('p','card-note','单模态候选按验证集选择 BrainGNN；+T1 单列为配对消融，其增益仍需检验。测试列用于冻结方案后的最终报告，不用于重新挑选模型。模拟分数不来自 12 人样例。'));append(el);await sleep(450);
    } else if(event.kind==='validation') {
      const el=table(['假设','验证指标（模拟）','结论'],event.rows,{title:'假设验证',note:'预设验证剧情，非科学证据'});el.dataset.kind='validation';append(el);await sleep(500);
    } else if(event.kind==='round') {
      const el=node('section','round');el.dataset.kind='round';el.dataset.round=event.round;
      const header=node('div','round-header','循环 '+event.round+' / 20');header.append(node('span','','目标 3 个'));el.append(header,node('p','',event.plan));
      const phases=node('div','round-steps');event.phases.forEach(p=>phases.append(node('span','',p)));el.append(phases);append(el);
      for(const phase of phases.children){$('status').textContent=`第 ${event.round} 轮 · ${phase.textContent}（模拟）`;phase.className='active';await sleep(event.delay);phase.className='done';}
      el.append(node('p','',event.finding));
      const reason=event.verdict.reason==='target_reached'?'达到开发集目标 → 冻结方案并进行最终验证':event.verdict.reason==='max_rounds'?'达到 20 轮上限 → 停止，保留未验证状态':'数量不足 → 进入下一轮';
      el.append(node('p','round-verdict',`当前不同假设 ${event.verdict.count} / 3 · ${reason}`));stick();
    }
    state.events.push(event);
  }
  function download(name,text) {
    const blob=new Blob([text],{type:'text/plain;charset=utf-8'}),url=URL.createObjectURL(blob),a=node('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function preview(name,text,allowDownload=true) {
    state.preview={name,text};$('preview-title').textContent=name;$('preview-content').textContent=text;$('download-artifact').hidden=!allowDownload;$('preview').showModal();
  }
  function populateArtifacts() {
    const entries=Object.entries(state.files);$('artifacts').replaceChildren();$('artifact-count').textContent=entries.length;
    if(!entries.length)$('artifacts').append(node('p','muted','普通聊天可使用右上角导出完整记录。'));
    entries.forEach(([name,text])=>{const b=node('button','artifact-btn',name);b.type='button';b.append(node('span','','预览 ↗'));b.onclick=()=>preview(name,text);$('artifacts').append(b);});
  }
  async function start() {
    if(state.controller)return;
    const options={fullOutcome:$('full-outcome').value},scenario=D.buildScenario(state.selected,options);
    if($('prompt').value.trim()!==scenario.prompt.trim()){
      $('status').textContent='此离线版本仅播放预设问题；请点击「恢复预设」后发送。';return;
    }
    const controller=new AbortController();state.controller=controller;state.completed=false;state.events=[];state.files={};
    $('welcome').hidden=true;$('messages').replaceChildren();$('artifacts').replaceChildren(node('p','muted','运行完成后生成文件预览。'));$('artifact-count').textContent='0';$('export').disabled=true;follow=true;
    append(node('div','user-message',scenario.prompt));const label=node('div','assistant-label');label.append(node('span','assistant-avatar','✦'),node('span','','NeuroDiscovery'),node('span','demo-badge','DEMO'));append(label);
    $('progress').style.width='0%';busy(true);
    try {
      await P.play(scenario.events,{signal:controller.signal,onEvent:render,onProgress:(done,total)=>{$('progress').style.width=(done/total*100)+'%';$('progress-label').textContent=done+' / '+total;}});
      if(state.controller!==controller)return;
      state.completed=true;state.files=D.artifacts(state.selected,options);populateArtifacts();
      $('status').textContent='演示完成 · 可查看产物，或再次发送重播';$('export').disabled=false;
      state.record={schema_version:1,distribution:'offline-demo',simulated:true,scientific_validation:false,model:'GPT-6-Astra',scenario:state.selected,mode:scenario.mode,prompt:scenario.prompt,options,events:state.events,artifacts:state.files};
    } catch(error) {
      if(error.name!=='AbortError'){$('status').textContent='播放失败：'+error.message;console.error(error);}
    } finally {if(state.controller===controller){state.controller=null;busy(false);}}
  }
  D.definitions.forEach((def,i)=>{const b=node('button','scene-button');b.type='button';b.dataset.scene=def.id;b.append(node('span','scene-icon',def.icon));const text=node('span','scene-copy');text.append(node('strong','',def.name),node('small','',def.subtitle));b.append(text,node('span','scene-index',String(i+1).padStart(2,'0')));b.onclick=()=>select(def.id);$('scenarios').append(b);});
  $('mode').onchange=()=>select($('mode').value);$('send').onclick=start;$('stop').onclick=stop;
  $('reset-prompt').onclick=()=>{$('prompt').value=D.definitions.find(d=>d.id===state.selected).prompt;$('status').textContent='已恢复预设问题';};
  $('prompt').onkeydown=e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();start();}};
  $('full-outcome').onchange=()=>select('full');
  $('export').onclick=()=>{if(state.completed)download('NeuroDiscovery-demo-'+state.selected+'.json',JSON.stringify(state.record,null,2));};
  $('theme').onclick=()=>{const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';$('theme').querySelector('span').textContent=dark?'切换浅色':'切换深色';};
  $('about').onclick=()=>preview('演示说明',D.disclosure+'\n\n五个场景沿用项目的模式边界、模型输入契约与三视角评审设计。播放器离线运行；不启动研究后端，不读取 API key。数据目录含可加载的合成数组；演示评分和训练曲线为预设值。\n\n步骤等待 0.5–1.6 秒，文字分块流式显示。Full 默认 3 轮达标，另可展示 20 轮达上限。\n\n/data/demo、/ideas 和 /experiments 是演示工作空间路径。桌面版可打开随包数据；IDEA.md、代码和结果可在播放完成后导出。\n\n普通聊天展示所选模型名 GPT-6-Astra，但没有执行该模型推理。',false);
  $('close-preview').onclick=()=>$('preview').close();$('download-artifact').onclick=()=>download(state.preview.name,state.preview.text);
  $('preview').addEventListener('click',e=>{if(e.target===$('preview')){const r=$('preview').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('preview').close();}});
  if(window.demoDesktop){$('open-data').hidden=false;$('open-data').onclick=async()=>{try{await window.demoDesktop.openData();}catch(e){$('status').textContent='无法打开数据目录：'+e.message;}};}
  // Read-only state for accessibility/smoke checks; execution uses the same user controls.
  window.demoStatus=()=>({scenario:state.selected,running:!!state.controller,completed:state.completed,events:state.events.length,artifacts:Object.keys(state.files)});
  select('chat');
})();
