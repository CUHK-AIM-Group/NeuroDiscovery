// Demo transport behind the unchanged production frontend. No LLM/graph/ML executor.
const http=require('node:http');
const fs=require('node:fs');
const path=require('node:path');
const {createHash}=require('node:crypto');
const D=require('./scenarios.js');
const {setTimeout:delay}=require('node:timers/promises');

const presets=D.definitions.map(({id,mode,prompt})=>({id,mode:mode==='end_to_end'?'end-to-end':mode,prompt}));
const limitSuffixes=['If the evidence remains insufficient, show how the system continues and eventually stops.','如果证据一直不足，请展示系统如何继续尝试并最终停止。','请演示持续证据不足、达到 20 轮上限的情况。'];
const limitPrompt=presets.find(p=>p.id==='full').prompt+' '+limitSuffixes[0];
const normalize=s=>String(s||'').normalize('NFKC').replace(/\s+/g,'').replace(/[，。！？；：“”‘’、,.!?;:"']/g,'').toLowerCase();
const acceptedPrompts=id=>{const definition=D.definitions.find(d=>d.id===id);return [definition.prompt,...definition.aliases||[]];};
function matchPrompt(text){
  const value=normalize(text);
  if(acceptedPrompts('full').some(prompt=>limitSuffixes.some(suffix=>normalize(prompt+suffix)===value)))return {...presets.find(p=>p.id==='full'),options:{fullOutcome:'limit'}};
  return presets.find(p=>acceptedPrompts(p.id).some(prompt=>normalize(prompt)===value));
}
function conversationTitle(userText){
  const preset=matchPrompt(userText);
  if(preset)return D.definitions.find(definition=>definition.id===preset.id).titles.en;
  const cleaned=String(userText||'').replace(/\s+/gu,' ').trim();
  const title=/[\u3400-\u9fff]/u.test(cleaned)?Array.from(cleaned).slice(0,28).join(''):Array.from(cleaned.split(/\s+/u).slice(0,8).join(' ')).slice(0,64).join('');
  return title.replace(/[.,!?;:，。！？；：]+$/u,'').trim()||'Research conversation';
}
function markdownTable(headers,rows){
  const cell=v=>String(v).replaceAll('|','\\|').replaceAll('\n',' ');
  return [headers.map(cell).join(' | '),headers.map(()=>'---').join(' | '),...rows.map(r=>r.map(cell).join(' | '))].map(r=>'| '+r+' |').join('\n');
}
function scores(rows=D.hypotheses){return markdownTable(['Rank','Hypothesis','Novelty','Structure','GNN','Statistical','Clinical','Methodological','Review mean','Total','Status'],rows.map((h,i)=>[`${i+1} / ${h.id}`,h.text,h.novelty,h.structural,h.gnn,h.statistical,h.clinical,h.methodological,h.review.toFixed(2),h.total.toFixed(2),h.status.replace(' (simulated)','')]));}
const graphSource=path.join(__dirname,'../../core/web/static/evidence-graph.js');
const Graph=require(fs.existsSync(graphSource)?graphSource:path.join(__dirname,'native-ui/evidence-graph.js'));
// Exact former illustrations only: restyle old conversations without rewriting receipts.
const LEGACY_GRAPH_SHA256=new Set(['42c7689db0b00a71db050bf5122fdc60cb3d2d64769b16ce0ffbed9c2233560f','b3adf97608f8546e54eeed421e267ae1f81781d6dda996ecd7077a9d4cd3b82d']);
function graphSvg(){
  // Coordinates are presentation only; the scenario still owns every concept/relation.
  const layout={
    adhd:{x:170,y:260,r:24,type:'Study population',color:'#3976ed'},
    cerebellum:{x:400,y:210,r:32,type:'Brain region',color:'#8b68d8'},
    dmn:{x:644,y:310,r:32,type:'Functional network',color:'#20a58c'},
    attention:{x:844,y:208,r:24,type:'Clinical phenotype',color:'#d9708f'},
    t1:{x:295,y:437,r:19,type:'Imaging measure',color:'#d8a34a'},
    motion:{x:795,y:447,r:19,type:'Study conditions',color:'#8294ad'},
  };
  const routes={
    'adhd>cerebellum':[265,180,272,208],
    'cerebellum>dmn':[545,182,544,222],
    'dmn>attention':[780,315,797,329],
    'adhd>attention':[460,22,489,130],
    't1>cerebellum':[350,365,349,341],
    'motion>dmn':[673,458,688,410],
  };
  return Graph.renderSvg({kind:Graph.kind,demo:true,scientific_validation:false,rendering:'evidence-map-v3',language:'en',title:'Cerebellar–default mode connectivity and attention symptoms',
    nodes:D.graph.nodes.map(([id,name])=>({id,name,...layout[id]})),edges:D.graph.edges,routes:D.graph.edges.map(([a,b])=>routes[a+'>'+b])});
}
const harness={schema_version:1,runtime_contract:'autoresearch-v1',runtime_features:{server_queue:true,tool_approval:true},capabilities:[
  {id:'chat',label:'Research chat',kind:'view',view:'chat'},
  {id:'graph',label:'Knowledge graph',kind:'view',view:'neurooracle'},
  {id:'autoresearch',label:'AutoResearch',kind:'configure'},
  {id:'evaluation',label:'Human Evaluation',kind:'view',view:'expert-study'},
  {id:'ranking',label:'Hypothesis ranking',kind:'view',view:'hypothesis-ranking'},
  {id:'results',label:'Evaluation results',kind:'view',view:'study-results'},
]};
async function createDemoServer({staticRoot,profile,workspace,timeScale=1}={}){
  fs.mkdirSync(profile,{recursive:true});
  const historyFile=path.join(profile,'native-history.json');
  let history={revision:0,state:null};
  if(fs.existsSync(historyFile))history=JSON.parse(fs.readFileSync(historyFile,'utf8'));
  // Translate only recognized auto-generated titles from previous builds.
  let renamed=false;
  for(const session of history.state?.sessions||[]){
    const firstUser=session.messages?.find(message=>message.role==='user');
    const preset=matchPrompt(firstUser?.content);
    const definition=D.definitions.find(item=>item.id===preset?.id);
    if(!preset||![preset.id,definition.titles.zh].includes(session.title))continue;
    session.title=conversationTitle(firstUser.content);
    session.autoTitlePending=false;renamed=true;
  }
  if(renamed){
    const backup=path.join(profile,'native-history-before-english-titles.json');
    if(!fs.existsSync(backup))fs.copyFileSync(historyFile,backup,fs.constants.COPYFILE_EXCL);
    history.revision++;
    fs.writeFileSync(historyFile+'.pending',JSON.stringify(history));fs.renameSync(historyFile+'.pending',historyFile);
  }
  const artifactRoot=path.join(profile,'native-artifacts');
  function saveAssets(run,assets){
    Object.assign(run.assets,assets);
    fs.mkdirSync(artifactRoot,{recursive:true});
    const target=path.join(artifactRoot,run.id+'.json');
    fs.writeFileSync(target+'.pending',JSON.stringify(run.assets));fs.renameSync(target+'.pending',target);
  }
  function savedAssets(id){
    const target=path.join(artifactRoot,id+'.json');
    if(fs.existsSync(target))return JSON.parse(fs.readFileSync(target,'utf8'));
    // Older builds saved the message but kept this fixed illustration only in RAM.
    const references=[`![Evidence subgraph](/demo/artifacts/${id}/evidence.svg)`,`![证据子图](/demo/artifacts/${id}/evidence.svg)`];
    const retained=(history.state?.sessions||[]).some(session=>(session.messages||[]).some(message=>
      message.role==='assistant'&&message.execution?.request_id===id&&references.some(reference=>String(message.content||'').includes(reference))));
    if(!retained)return {};
    const recovered={id,assets:{}};saveAssets(recovered,{'evidence.svg':graphSvg()});return recovered.assets;
  }
  const runs=new Map(),requests=[];let origin,closing=false;
  const modelEnv=()=>({provider:'local',model:'GPT-6-Astra',api_key_required:false,api_key_present:null,available_models:[{provider:'local',model:'GPT-6-Astra'}]});
  const snapshot=run=>({request_id:run.id,chat_id:run.chatId,seq:run.seq,status:run.status,blocks:run.blocks,tools:run.tools});
  function touch(run){run.seq++;}
  async function pause(ms,run){
    if(ms<0||ms>5000)throw new Error('Step wait exceeds limit');
    run.maxDelay=Math.max(run.maxDelay,ms);
    const scale=typeof timeScale==='function'?timeScale(run):timeScale;
    await delay(ms*scale,null,{signal:run.controller.signal});
  }
  async function text(run,value){
    const block={call_id:'call-'+run.blocks.length,model:'GPT-6-Astra',source:'main',status:'running',reasoning:'Plan: '+value.slice(0,85),text:''};
    run.blocks.push(block);touch(run);await pause(180,run);
    const chars=Array.from(value);
    for(let i=0;i<chars.length;i+=48){await pause(35,run);block.text+=chars.slice(i,i+48).join('');touch(run);}
    block.status='completed';touch(run);
  }
  async function tool(run,name,input,output,ms=900){
    const item={tool_id:'tool-'+run.tools.length,tool:name,status:'running',command:JSON.stringify(input,null,2),output:''};
    run.tools.push(item);touch(run);await pause(ms,run);item.output=output;item.status='completed';touch(run);
  }
  function completed(run,status){
    for(const block of run.blocks)if(block.status==='running')block.status=status;
    for(const item of run.tools)if(item.status==='running')item.status=status;
    run.status=status;touch(run);
    run.result={type:'message',content:run.blocks.map(b=>b.text).join('\n\n'),model_used:'GPT-6-Astra',autoresearch_mode:run.mode,execution:snapshot(run),demo:true,scientific_validation:false};
    if(!closing&&![...runs.values()].some(r=>r.chatId===run.chatId&&r.status==='running')){
      const next=[...runs.values()].find(r=>r.chatId===run.chatId&&r.status==='queued');if(next)void execute(next);
    }
  }
  async function execute(run){
    run.status='running';touch(run);
    try {
      const preset=matchPrompt(run.message);
      if(!preset){await text(run,'Choose one of these five tasks, set the corresponding AutoResearch mode, and send its prompt:\n\n'+presets.map(p=>`**${p.id} · ${p.mode}**\n\n${p.prompt}`).join('\n\n'));completed(run,'completed');return;}
      if(run.mode!==preset.mode){await text(run,`This request uses ${preset.mode==='model'?'Model (Experiment)':preset.mode==='end-to-end'?'Full':preset.mode==='off'?'Off':preset.mode} mode. Switch the AutoResearch scope in the model menu at the bottom right of the composer, then send the same prompt.`);completed(run,'completed');return;}
      run.scenario=preset.id;run.options=preset.options||{};
      const completedRounds=[];
      for(const event of D.buildScenario(preset.id,run.options).events){
        switch(event.kind){
          case 'message':await text(run,event.text);break;
          case 'tool':await tool(run,event.name,event.input,event.output,event.delay);break;
          case 'graph':saveAssets(run,{'evidence.svg':graphSvg()});await text(run,`![Evidence subgraph](/demo/artifacts/${run.id}/evidence.svg)\n\nRetain the study population, measurements and confound controls. Adjacent graph relations do not by themselves establish a novel finding.`);break;
          case 'scores': {
            const pool=event.pool||D.hypothesisPool,name=event.artifact||'hypotheses.csv';
            saveAssets(run,{[name]:D.hypothesisCsv(pool)});
            const link=`[Open the complete hypothesis table: ${name} (${pool.length} rows, CSV)](/demo/artifacts/${run.id}/${name})`;
            await text(run,`**${event.title||'Hypothesis ranking'}**\n\n${event.summary||''}\n\n${link}\n\n${scores(event.rows)}\n\n${D.scoreNote}`);break;
          }
          case 'datasets':for(const d of D.datasets)await text(run,`**${d.name} · ${d.modalities}**\n\nInput state: ${d.downloaded}\n\nPreparation: ${d.pipeline}\n\nModel inputs: ${d.ready} → ${d.models}\n\n[Official documentation](${d.source})`);break;
          case 'tree':await text(run,'**Final dataset layout**\n\n```text\n'+event.text+'\n```');break;
          case 'code':await text(run,`**${event.filename}**\n\n\`\`\`${event.filename.endsWith('.py')?'python':'json'}\n${event.text}\n\`\`\``);break;
          case 'capabilities':await text(run,markdownTable(['Mode','Scope'],[['Off','Research conversation'],['Idea','Evidence, hypotheses and review'],['Data','Preparation, quality control and splits'],['Model','Model implementation, tuning and experiments'],['Full','Connect all stages and iterate on validation feedback']]));break;
          case 'training': {
            const steps=[.53,.621,.693,.754,.786,.809];
            for(let i=0;i<steps.length;i++)await tool(run,`Training · Epoch ${(i+1)*5}/30`,{validation_only:true},`Validation AUROC = ${steps[i].toFixed(3)}; final test set unopened.`,500);
            await text(run,markdownTable(['Epoch','Validation AUROC'],steps.map((v,i)=>[(i+1)*5,v.toFixed(3)])));break;
          }
          case 'results':await text(run,'**Experiment results**\n\n'+markdownTable(D.resultHeaders,D.results)+'\n\nSingle-modality models are selected on validation data; +T1 is a paired ablation. Final test metrics are reported once, after the protocol is frozen.');break;
          case 'validation':await text(run,'**Hypothesis validation**\n\n'+markdownTable(['Hypothesis','Evidence','Conclusion'],event.rows));break;
          case 'round': {
            await text(run,`**Round ${event.round} experiment · ${event.title}**\n\n${event.plan}`);
            const feedback={Idea:event.plan,Data:event.dataReview,Experiment:event.experiment,Review:event.finding};
            for(const phase of event.phases) {
              await tool(run,`Round ${event.round}/20 · ${phase}`,{round:event.round,phase,evaluation_split:'development'},feedback[phase],event.delay);
              if(phase==='Experiment')await text(run,`**Round ${event.round} results (development)**\n\n${markdownTable(D.roundHeaders,event.comparisons)}`);
            }
            completedRounds.push(event);
            saveAssets(run,D.roundArtifacts(completedRounds));
            const filename=`round_${String(event.round).padStart(2,'0')}.csv`;
            await text(run,D.roundDecision(event)+`\n\n[Open the round ${event.round} results](/demo/artifacts/${run.id}/${filename})`);break;
          }
        }
      }
      const artifacts=D.artifacts(preset.id,run.options);if(Object.keys(artifacts).length)saveAssets(run,artifacts);
      if(Object.keys(artifacts).length)await text(run,'**Output files**\n\n'+Object.keys(artifacts).map(name=>`- [${name}](/demo/artifacts/${run.id}/${name})`).join('\n'));
      completed(run,'completed');
    }catch(error){
      if(error.name!=='AbortError')run.blocks.push({call_id:'error',model:'GPT-6-Astra',status:'failed',text:'Request failed: '+error.message});
      completed(run,error.name==='AbortError'?'cancelled':'failed');
    }
  }
  // Decode the request body once, after every chunk has arrived. Decoding each
  // chunk separately corrupts any multi-byte character that a TCP boundary
  // splits, which silently damaged saved conversation history.
  async function body(req){const chunks=[];let bytes=0;for await(const chunk of req){bytes+=chunk.length;if(bytes>12*1024*1024)throw new Error('Request too large');chunks.push(chunk);}const text=Buffer.concat(chunks).toString('utf8');return text?JSON.parse(text):{};}
  const server=http.createServer(async(req,res)=>{
    const json=(value,status=200)=>{res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify(value));};
    const file=(target)=>{
      if(!fs.existsSync(target)||!fs.statSync(target).isFile())return json({message:'Not found'},404);
      const types={'.html':'text/html','.css':'text/css','.js':'application/javascript','.json':'application/json','.png':'image/png','.svg':'image/svg+xml'};
      res.writeHead(200,{'Content-Type':(types[path.extname(target)]||'application/octet-stream')+'; charset=utf-8','Cache-Control':'no-store'});fs.createReadStream(target).pipe(res);
    };
    try{
      if(req.headers.origin&&req.headers.origin!==origin)return json({message:'Application origin required'},403);
      const url=new URL(req.url,origin),pathname=decodeURIComponent(url.pathname);requests.push(req.method+' '+pathname);
      if(pathname==='/api/harness')return json(harness);
      if(pathname==='/api/distribution/demo')return json({distribution:'native-ui-demo',live_calls:false});
      if(pathname==='/api/env'||pathname==='/api/env/model')return json(modelEnv());
      if(pathname==='/api/skills')return json({skills:[]});
      if(pathname==='/api/llm/composer-capabilities')return json({reasoning_efforts:['low','medium','high']});
      if(pathname==='/api/checkpoints')return json({checkpoints:[]});
      if(pathname==='/api/neurooracle/graph/status')return json({available:false,reason:'See the evidence subgraph in the current conversation.'});
      if(pathname==='/api/workbench/validate-workspace')return json({path:workspace});
      if(pathname==='/api/workbench/usage')return json({totals:{requests:0,input_tokens:0,output_tokens:0,known_fields:{}},rows:[]});
      if(pathname==='/api/workbench/state'){
        if(req.method==='PUT'){
          const value=await body(req);if(value.revision!==history.revision)return json({message:'History revision conflict'},409);
          const next={revision:history.revision+1,state:value.state};
          fs.writeFileSync(historyFile+'.pending',JSON.stringify(next));fs.renameSync(historyFile+'.pending',historyFile);history=next;
        }return json(history);
      }
      if(pathname==='/api/chat/title'){
        const value=await body(req),content=String(value.user||'').trim()||String(value.assistant||'').trim();
        if(!content)return json({type:'error',message:'Missing conversation content'},400);
        return json({type:'done',title:conversationTitle(content,value.language)});
      }
      if(pathname==='/api/chat/tasks'||pathname.startsWith('/api/chat/tasks/'))return json({items:[]});
      if(pathname.startsWith('/api/chat/queue/'))return json({items:[...runs.values()].filter(r=>r.chatId===pathname.split('/')[4]).map(r=>({request_id:r.id,message:r.message,status:r.status,result:r.result||null}))});
      if(pathname==='/api/chat/cancel'){
        const value=await body(req),run=runs.get(value.request_id);if(run){if(run.status==='queued')completed(run,'cancelled');else if(run.status==='running')run.controller.abort();}return json({ok:true});
      }
    if(pathname==='/api/chat/steer'||pathname==='/api/chat/compact')return json({message:'Stop the current task before sending a new request.'},400);
      if(pathname==='/api/chat'&&req.method==='POST'){
        const value=await body(req);
        if(!/^[\w-]{1,100}$/.test(value.request_id||'')||!value.chat_id)return json({message:'Invalid request identity'},400);
        let run=runs.get(value.request_id);
        if(run&&(run.chatId!==value.chat_id||run.message!==value.message||run.mode!==(value.autoresearch_mode||'off')))return json({message:'Request identity conflict'},409);
        if(!run){
          run={id:value.request_id,chatId:value.chat_id,message:String(value.message||''),mode:value.autoresearch_mode||'off',seq:0,status:'queued',blocks:[],tools:[],assets:{},controller:new AbortController(),maxDelay:0};runs.set(run.id,run);
          if(![...runs.values()].some(r=>r!==run&&r.chatId===run.chatId&&r.status==='running'))void execute(run);
        }
        return json({type:'accepted',request_id:run.id,chat_id:run.chatId,autoresearch_mode:run.mode,status:run.status});
      }
      if(pathname.startsWith('/api/chat/runs/')){
        const run=runs.get(pathname.split('/').at(-1));if(!run)return json({message:'Run not found'},404);
        return json({snapshot:snapshot(run),events:[],status:run.status,seq:run.seq,result:run.result||null});
      }
      if(pathname.startsWith('/demo/artifacts/')){
        const parts=pathname.split('/'),id=parts[3],name=parts[4];
        if(parts.length!==5||!/^\w[\w-]{0,99}$/.test(id)||!/^[\w][\w.-]{0,99}$/.test(name))return json({message:'Artifact not found'},404);
        const assets=runs.get(id)?.assets||savedAssets(id);
        if(!Object.hasOwn(assets,name))return json({message:'Artifact not found'},404);
        let content=assets[name];
        if(name==='evidence.svg'&&typeof content==='string'&&LEGACY_GRAPH_SHA256.has(createHash('sha256').update(content).digest('hex')))content=graphSvg();
        const svg=name.endsWith('.svg');res.writeHead(200,{'Content-Type':svg?'image/svg+xml':name.endsWith('.csv')?'text/csv; charset=utf-8':'text/plain; charset=utf-8','Content-Length':Buffer.byteLength(content),'Cache-Control':'no-store',...(svg?{}:{'Content-Disposition':`attachment; filename="${name}"`})});res.end(req.method==='HEAD'?undefined:content);return;
      }
      if(pathname==='/harness'||pathname==='/')return file(path.join(staticRoot,'index.html'));
      const routes={'/neurooracle':'explore.html','/explore':'explore.html','/study':'study.html','/discovery-study':'discovery-study.html','/evaluation':'evaluation-home.html'};
      if(routes[pathname])return file(path.join(staticRoot,routes[pathname]));
      if(pathname.startsWith('/vendor/')){
        const name=path.basename(pathname);if(!['marked.min.js','highlight.min.js','github.min.css'].includes(name))return json({message:'Not found'},404);return file(path.join(__dirname,'vendor',name));
      }
      if(pathname.startsWith('/static/')){
        const target=path.resolve(staticRoot,pathname.slice(8));if(!target.startsWith(path.resolve(staticRoot)+path.sep))return json({message:'Forbidden'},403);return file(target);
      }
      return json({message:'This operation is currently unavailable.'},404);
    }catch(error){if(!res.headersSent)json({message:error.message},500);else res.end();}
  });
  // Windows can allocate an ephemeral port that browsers reject (for example 6667).
  const unsafePorts=new Set([1,7,9,11,13,15,17,19,20,21,22,23,25,37,42,43,53,69,77,79,87,95,101,102,103,104,109,110,111,113,115,117,119,123,135,137,139,143,161,179,389,427,465,512,513,514,515,526,530,531,532,540,548,554,556,563,587,601,636,989,990,993,995,1719,1720,1723,2049,3659,4045,5060,5061,6000,6566,6665,6666,6667,6668,6669,6697,10080]);
  for(let attempt=0;attempt<20;attempt++){
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
    if(!unsafePorts.has(server.address().port))break;
    await new Promise(resolve=>server.close(resolve));
    if(attempt===19)throw Error('Could not allocate a browser-compatible local port.');
  }
  origin='http://127.0.0.1:'+server.address().port;
  return {server,origin,runs,requests,modelEnv,async close(){if(closing)return;closing=true;for(const r of runs.values())if(r.status==='running')r.controller.abort();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}};
}
module.exports={createDemoServer,presets,limitPrompt,matchPrompt,harness};
