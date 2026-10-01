const test = require('node:test');
const assert = require('node:assert/strict');
const W = require('../client-workbench.js');
const Commands = require('../composer-commands.js');

test('chat list has no Recent heading while preserving pinned, archived and pagination',()=>{
  const fs=require('node:fs'), path=require('node:path'), vm=require('node:vm');
  const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
  const source=html.slice(html.indexOf('function renderChatList('),html.indexOf('function formatSkillName('));
  for (const zh of [false,true]) {
    const sessions=[{id:'pinned',pinned:true},{id:'first'},{id:'second'}];
    const context={getUnassignedSessions:()=>sessions,state:{sessions:[...sessions,{id:'old',archived:true}],workbenchChatLimit:1},
      chatListEl:{innerHTML:''},escapeHtml:value=>value,uiText:(en,cn)=>zh?cn:en,tr:value=>value,
      renderSessionListItem:session=>`<li>${session.id}</li>`};
    vm.createContext(context);vm.runInContext(source,context);context.renderChatList();
    const rendered=context.chatListEl.innerHTML;
    assert.doesNotMatch(rendered,/Recent|最近/);
    assert.ok(rendered.includes(zh?'置顶':'Pinned'));
    assert.ok(rendered.includes(zh?'已归档':'Archived'));
    assert.ok(rendered.includes('<li>first</li>'));
    assert.ok(!rendered.includes('<li>second</li>'));
    assert.ok(rendered.includes('data-wb-more'));
    sessions.splice(0);
    context.state.sessions=[];context.renderChatList();
    assert.equal(context.chatListEl.innerHTML,'');
  }
});

test('composer omits interrupted recovery banners without dismissing saved requests',()=>{
  const fs=require('node:fs'), path=require('node:path'), vm=require('node:vm');
  const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
  const source=html.slice(html.indexOf('function renderMessageQueue('),html.indexOf('async function sendMessage('));
  const element=()=>({children:[],hidden:false,append(...items){this.children.push(...items);},
    replaceChildren(){this.children=[];},get childElementCount(){return this.children.length;}});
  const panel=element(), interrupted={status:'interrupted',message:'Saved research',request_id:'interrupted'};
  const session={serverQueue:[interrupted],messageQueue:[]};
  const context={document:{getElementById:()=>panel,createElement:element},getActiveSession:()=>session,
    isSessionWaiting:()=>false,inputEl:{value:''},uiText:en=>en,
    fetch:()=>{throw new Error('Rendering must not cancel or recover requests');}};
  vm.createContext(context);vm.runInContext(source,context);
  context.renderMessageQueue();
  assert.equal(panel.childElementCount,0);
  assert.equal(panel.hidden,true);
  assert.equal(session.serverQueue[0],interrupted);
  session.serverQueue.push({status:'queued',message:'Next prompt',request_id:'queued'});
  context.renderMessageQueue();
  assert.equal(panel.hidden,false);
  assert.equal(panel.children[0].children[0].textContent,'queued · Next prompt');
  assert.equal(panel.children[1].textContent,'Resume server queue');
  assert.ok(html.includes('Resume AutoResearch from checkpoint'));
  const css=fs.readFileSync(path.join(__dirname,'../client-workbench.css'),'utf8');
  assert.match(css,/\.wb-message-queue\[hidden\]\s*\{\s*display:\s*none;/);
});

test('research slash commands select full and component modes only when explicitly entered',()=>{
  for(const [command,mode] of [['/autoresearch','end-to-end'],['/data','data'],['/model','model'],['/idea','idea'],['/autoresearch full','end-to-end'],['/autoresearch idea','idea']]) {
    assert.equal(Commands.parse(command).mode,mode);
    assert.equal(Commands.parse(command+' analyze this').text,'analyze this');
  }
  assert.equal(Commands.parse('/goal'),null);
  assert.equal(Commands.parse('Quoted /autoresearch analyze this'),null);
  assert.equal(Commands.parse('/ideas'),null);
  assert.equal(Commands.parse('/tasks').action,'tasks');
  assert.equal(Commands.parse('/usage').action,'usage');
});

test('bare research commands change mode without dispatch; panel commands never start jobs',async()=>{
  const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
  const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
  const source=html.slice(html.indexOf('async function handleComposerCommand('),html.indexOf('async function openScheduledTasks('));
  const session={researchMode:'off'},opened=[];
  const context={window:{ComposerCommands:Commands},state:{},inputEl:{value:'/autoresearch',focus(){}},
    syncModelTrigger(){},renderModelMenu(){},saveSessions(){},rememberComposerSettings(){},closeSlashMenu(){},setStatus(){},tr:value=>value,autoResearchLabel:mode=>mode,uiText:en=>en,
    setActiveView:view=>opened.push(view),openScheduledTasks:async value=>opened.push(value),compactConversation:()=>assert.fail('Unexpected compaction'),showError:message=>{throw Error(message);}};
  vm.createContext(context);vm.runInContext(source,context);
  assert.equal((await context.handleComposerCommand('/autoresearch',session)).handled,true);
  assert.equal(session.researchMode,'end-to-end');
  const command=await context.handleComposerCommand('/idea read papers',session);
  assert.equal(command.handled,false);assert.equal(command.text,'read papers');assert.equal(session.researchMode,'idea');
  await context.handleComposerCommand('/tasks',session);assert.equal(opened[0],session);
  await context.handleComposerCommand('/usage',session);assert.equal(context.state.settingsSection,'usage');assert.equal(opened[1],'settings');
});

test('AutoResearch preflight refuses an old backend before sending any research', async()=>{
  const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
  const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
  const source=html.slice(html.indexOf('async function performWorkbenchChat('),html.indexOf('async function resumeWorkbenchRequest('));
  const requests=[];
  const context={workbenchHistory:null,uiText:en=>en,fetch:async(url)=>{requests.push(url);return {ok:true,json:async()=>({schema_version:1})};},followWorkbenchRequest:()=>assert.fail('No run may start')};
  vm.createContext(context);vm.runInContext(source,context);
  const result=await context.performWorkbenchChat({autoresearch_mode:'idea'}, {}, {});
  assert.match(result.message,/outdated/);
  assert.deepEqual(requests,['/api/harness']);
});

test('request mode comes from its conversation, not a stale global menu',()=>{
  const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
  const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
  const source=html.slice(html.indexOf('function beginSessionRequest('),html.indexOf('function finishSessionRequest('));
  const state={sessions:[{id:'idea',researchMode:'idea'},{id:'ordinary',researchMode:'off'}],sessionRequests:{},autoResearchMode:'off'};
  const context={state,isSessionWaiting:()=>false,saveSessions(){},AUTO_RESEARCH_MODE_OFF:'off'};
  vm.createContext(context);vm.runInContext(source,context);
  assert.equal(context.beginSessionRequest('idea',{}),true);
  assert.equal(state.sessionRequests.idea.researchMode,'idea');
  assert.equal(context.beginSessionRequest('ordinary',{}),true);
  assert.equal(state.sessionRequests.ordinary.researchMode,'off');
});

test('chat defaults to high independently of old choices and respects provider capabilities',async()=>{
  const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
  const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
  const settings=html.slice(html.indexOf('async function composerReasoningEffort('),html.indexOf('function renderModelOptions('));
  const request=html.slice(html.indexOf('async function performWorkbenchChat('),html.indexOf('async function resumeWorkbenchRequest('));
  for(const [efforts,expected] of [[['low','medium','high'],'high'],[[],'default'],[['low','xhigh'],'default']]){
    const posted=[];
    const context={workbenchHistory:null,uiText:en=>en,currentModelChoice:()=>({provider:'fixture',model:'test'}),
      followWorkbenchRequest:async()=>({type:'done'}),fetch:async(url,options)=>{
        if(url==='/api/harness')return {ok:true,json:async()=>({runtime_contract:'autoresearch-v1',runtime_features:{server_queue:true,tool_approval:true}})};
        if(url.startsWith('/api/llm/composer-capabilities?'))return {ok:true,json:async()=>({reasoning_efforts:efforts})};
        assert.equal(url,'/api/chat');posted.push(JSON.parse(options.body));
        return {ok:true,json:async()=>({type:'accepted',autoresearch_mode:'off'})};
      }};
    vm.createContext(context);vm.runInContext(settings+request,context);
    await context.performWorkbenchChat({autoresearch_mode:'off'},{reasoningEffort:'low'},{});
    assert.equal(posted.length,1);assert.equal(posted[0].reasoning_effort,expected);
  }
});

function queueFixture() {
  const fs=require('node:fs'), path=require('node:path'), vm=require('node:vm');
  const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
  const source=html.slice(html.indexOf('async function sendMessage('),html.indexOf('function applyTheme('));
  const session={id:'chat',messageQueue:[],messages:[],selectedSkills:[],draft:''};
  const requests=[],pending={}; let sequence=0;
  const context={session,requests,AbortController,Array,console,
    inputEl:{value:'queued prompt',style:{}}, state:{pendingFiles:[],autoResearchMode:'data',sessions:[session],activeSessionId:'chat',selectedSkills:[]},
    getActiveSession:()=>session, getSessionSelectedSkills:()=>session.selectedSkills,
    isSessionWaiting:()=>Boolean(pending.request), ensureChatLlmConfigured:async()=>true,
    getHypothesisMode:()=> 'novelty_first', composerReasoningEffort:async()=> 'high', buildSkillInstruction:()=>'',buildAttachmentText:()=>'',
    uiText:en=>en, tr:text=>text, newMessageId:()=>`message-${++sequence}`, nowISO:()=> 'now',
    workspacePathForSession:()=>'/workspace', clearPendingFiles:()=>{context.state.pendingFiles=[];},
    serverQueueAvailable:()=>true,
    saveSessions(){}, renderMessageQueue(){}, chatPanelEl:{classList:{remove(){}}}, welcomeEl:{style:{}},
    beginAutoTitleForFirstUserTurn(){}, beginSessionRequest:()=>{if(pending.request)return false; pending.request={requestId:`request-${++sequence}`,researchMode:'off'};session.pendingRequest={};return true;},
    sessionRequestFor:()=>pending.request, renderSidebarLists(){},renderActiveSession(){},loadCheckpoints(){},scrollToBottom(){},closeSlashMenu(){},currentUiLanguage:()=> 'en',
    performWorkbenchChat:async payload=>{requests.push(payload);return {type:'done',content:'Synthetic reply',execution:{status:'completed'}};},
    renderSessionProgress(){},renderSkills(){},renderPendingFiles(){},updateSessionTitleFromFirstTurn(){},
    finishSessionRequest:()=>{delete pending.request;},showError:message=>{throw new Error(message);},
  };
  vm.createContext(context); vm.runInContext(source,context);
  return {context,session,requests,pending};
}

test('busy composer submits durable server queue entries without starting client execution', async()=>{
  const {context,session,requests,pending}=queueFixture();
  pending.request={requestId:'busy'};
  session.reasoningEffort='low';
  const posted=[];
  context.crypto=require('node:crypto').webcrypto;
  context.fetch=async(url, options)=>{posted.push(JSON.parse(options.body));return {ok:true,json:async()=>({type:'accepted'})};};
  context.refreshServerQueue=async()=>{};
  await context.sendMessage();
  context.inputEl.value='second queued prompt'; await context.sendMessage();
  assert.equal(requests.length,0); assert.equal(session.messageQueue.length,0);
  assert.deepEqual(posted.map(item=>item.message),['queued prompt','second queued prompt']);
  assert.ok(posted.every(item=>item.server_queue===true));
  assert.ok(posted.every(item=>item.reasoning_effort==='high'));
  assert.notEqual(posted[0].request_id,posted[1].request_id);
});

test('a backend without the queue endpoint keeps the queue in the browser', async()=>{
  const {context,session,requests,pending}=queueFixture();
  context.serverQueueAvailable=()=>false;
  pending.request={requestId:'busy'};
  const posted=[];
  context.crypto=require('node:crypto').webcrypto;
  context.fetch=async(url, options)=>{posted.push(url);return {ok:true,json:async()=>({type:'accepted'})};};
  context.inputEl.value='held locally';
  await context.sendMessage();
  assert.equal(posted.length,0);
  assert.equal(requests.length,0);
  assert.equal(session.messageQueue.length,1);
  assert.equal(session.messageQueue[0].text,'held locally');
});

test('/compact is a local control and never becomes a model prompt',async()=>{
  const {context,session,requests}=queueFixture();
  let compacted=null;context.compactConversation=async value=>{compacted=value;};
  context.inputEl.value='/compact';
  context.ensureChatLlmConfigured=()=>assert.fail('Compaction must not require a model');
  await context.sendMessage();
  assert.equal(compacted,session);assert.equal(requests.length,0);assert.equal(session.messages.length,0);
});

test('cancellation pauses remaining queue and never dispatches another conversation', async()=>{
  const {context,session,requests,pending}=queueFixture();
  session.messageQueue=[{id:'first',text:'first',finalText:'first',displayText:'first',selectedSkills:[],noveltyMode:'novelty_first',researchMode:'data'},
    {id:'second',text:'remaining',finalText:'remaining',displayText:'remaining',selectedSkills:[],noveltyMode:'novelty_first',researchMode:'data'}];
  context.performWorkbenchChat=async payload=>{requests.push(payload);return {type:'done',content:'Stopped',execution:{status:'cancelled'}};};
  await context.sendMessage(session.messageQueue[0]);
  assert.equal(requests.length,1); assert.equal(session.messageQueue.length,1);
  context.ensureChatLlmConfigured=async()=>{context.getActiveSession=()=>({id:'other'});return true;};
  await context.sendMessage(session.messageQueue[0]);
  assert.equal(requests.length,1); assert.equal(session.messageQueue.length,1);
});

test('two simultaneous queue-drain actions dispatch an entry only once',async()=>{
  const {context,session,requests,pending}=queueFixture();
  session.messageQueue=[{id:'first',text:'first',finalText:'first',displayText:'first',selectedSkills:[],noveltyMode:'novelty_first',researchMode:'data'}];
  const item=session.messageQueue[0];
  await Promise.all([context.sendMessage(item),context.sendMessage(item)]);
  assert.equal(requests.length,1);
});

test('agent activity reflects real model/tool state and terminal states take precedence', () => {
  const execution = {status:'running', blocks:[], tools:[]};
  assert.equal(W.activityLabel(execution), 'Working');
  execution.blocks.push({status:'running'});
  assert.equal(W.activityLabel(execution, true), '等待模型响应');
  execution.blocks[0].reasoning = 'Provider summary';
  assert.equal(W.activityLabel(execution), 'Thinking…');
  execution.blocks[0].text = 'Response';
  assert.equal(W.activityLabel(execution, true), '正在生成回复…');
  assert.equal(W.activityLabel(execution), 'Writing a response…');
  execution.tools.push({status:'running', tool:'read_workspace_file'});
  assert.equal(W.activityLabel(execution), 'Using tool · read_workspace_file');
  for (const [status, label] of Object.entries({stopping:'停止中…', completed:'已完成', failed:'失败', cancelled:'已取消', interrupted:'中断', blocked:'阻塞'})) {
    execution.status = status;
    assert.equal(W.activityLabel(execution, true), label);
  }
});

function fixture() {return {projects:[{id:'a',workspacePath:'/a'},{id:'b',workspacePath:'/b'}],sessions:[{id:'s',projectId:'a',draft:'keep',messages:[{content:'keep'}],pinned:true}],activeSessionId:'s',activeProjectId:'a'};}
test('removing a workspace preserves chats, drafts, pins and original cwd',()=>{
  const state=fixture();W.removeWorkspace(state,'a',()=>false);
  assert.equal(state.sessions.length,1);assert.equal(state.sessions[0].workspacePath,'/a');assert.equal(state.sessions[0].projectId,null);
  assert.equal(state.sessions[0].messages[0].content,'keep');assert.equal(state.sessions[0].draft,'keep');assert.equal(state.sessions[0].pinned,true);
});
test('busy remove/move is atomic; moving retains identity and history',()=>{
  const state=fixture(),before=JSON.stringify(state);
  assert.throws(()=>W.removeWorkspace(state,'a',()=>true));assert.throws(()=>W.moveSession(state,'s','b',()=>true));assert.equal(JSON.stringify(state),before);
  W.moveSession(state,'s','b',()=>false);assert.equal(state.sessions[0].workspacePath,'/b');assert.equal(state.sessions[0].id,'s');
  W.moveSession(state,'s','',()=>false);assert.equal(state.sessions[0].workspacePath,'/b');assert.equal(state.activeProjectId,null);
});
test('duplicate, old and other-chat events cannot corrupt output',()=>{
  const state={request_id:'r',chat_id:'s',seq:0,blocks:[],tools:[]};
  const e={request_id:'r',chat_id:'s',seq:1,type:'text',call_id:'c',text:'hello'};
  W.reduceEvent(state,e);W.reduceEvent(state,e);W.reduceEvent(state,{...e,seq:2,chat_id:'other'});W.reduceEvent(state,{...e,seq:3,request_id:'other'});
  assert.equal(state.blocks[0].text,'hello');assert.equal(state.seq,1);
});
test('one upstream snapshot completes without reposting chat',async()=>{
  let count=0;
  const result=await W.followRun('r','s',{fetcher:async url=>{count++;assert.match(url,/^\/api\/chat\/runs\/r/);return {ok:true,json:async()=>({seq:1,status:'completed',snapshot:{request_id:'r',chat_id:'s',seq:1,status:'completed',blocks:[],tools:[]},result:{type:'done',content:'ok'}})};}});
  assert.equal(count,1);assert.equal(result.content,'ok');
});
test('missing request stops recovery without sending it again',async()=>{
  const result=await W.followRun('r','s',{fetcher:async()=>({ok:false,status:404,json:async()=>({message:'Missing'})})});
  assert.equal(result.type,'error');assert.equal(result.message,'Missing');
});
test('missing usage, real zero, partial coverage and CSV injection',()=>{
  assert.equal(W.countLabel({totals:{requests:1,known_fields:{input_tokens:0}}},'input_tokens'),'Not reported');
  assert.equal(W.countLabel({totals:{requests:1,input_tokens:0,known_fields:{input_tokens:1}}},'input_tokens'),'0');
  assert.match(W.countLabel({totals:{requests:2,input_tokens:4,known_fields:{input_tokens:1}}},'input_tokens'),/partial/);
  assert.ok(W.csv([{model:'=FORMULA()',input_tokens:null}]).includes('"\'=FORMULA()"'));
});
test('legacy state migrates once, and a revision conflict never overwrites',async()=>{
  const snapshot={sessions:[{id:'s',draft:'legacy'}],projects:[]},requests=[],errors=[];
  const history=W.createHistory({snapshot:()=>snapshot,restore:()=>assert.fail('No server state'),onError:e=>errors.push(e),fetcher:async(url,options)=>{
    requests.push(options);
    if(!options)return {ok:true,json:async()=>({revision:0,state:null})};
    if(requests.length===2)return {ok:true,json:async()=>({revision:1})};
    return {ok:false,status:409,json:async()=>({message:'Conflict'})};
  }});
  await history.load();assert.equal(JSON.parse(requests[1].body).state.sessions[0].draft,'legacy');
  history.schedule();await history.flush();assert.equal(history.conflict,true);history.schedule();await history.flush();
  assert.equal(requests.length,3);assert.equal(errors.length,1);
});
test('server state wins over a stale local backup on later opens',async()=>{
  let restored;
  const history=W.createHistory({snapshot:()=>assert.fail('No overwrite'),restore:s=>restored=s,onError:e=>{throw e;},fetcher:async()=>({ok:true,json:async()=>({revision:5,state:{sessions:[{id:'server'}],projects:[]}})})});
  await history.load();assert.equal(restored.sessions[0].id,'server');
});
test('live usage includes each model call once and keeps absent counters unknown',()=>{
  const execution={request_id:'r',chat_id:'s',seq:0,blocks:[],tools:[]};
  W.reduceEvent(execution,{request_id:'r',chat_id:'s',seq:1,type:'model_start',call_id:'a'});
  W.reduceEvent(execution,{request_id:'r',chat_id:'s',seq:2,type:'usage',call_id:'a',usage:{input_tokens:12,output_tokens:2}});
  W.reduceEvent(execution,{request_id:'r',chat_id:'s',seq:3,type:'model_start',call_id:'b'});
  const totals=W.executionUsage(execution).totals;
  assert.equal(totals.requests,2);assert.equal(totals.input_tokens,12);assert.equal(totals.known_fields.input_tokens,1);
  assert.equal(W.countLabel({totals},'cache_read_tokens'),'Not reported');
});

test('completed turns preserve reading position and cannot switch another chat',()=>{
  const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
  const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
  const source=html.slice(html.indexOf('function renderSessionProgress('),html.indexOf('function beginEditingUserMessage('));
  const calls=[],context={state:{activeSessionId:'s'},messagesEl:{scrollHeight:3000,scrollTop:100,clientHeight:600},renderActiveSession:options=>calls.push(options)};
  vm.createContext(context);vm.runInContext(source,context);
  context.renderSessionProgress('other');assert.equal(calls.length,0);
  context.renderSessionProgress('s');assert.equal(calls[0].preserveScroll,true);
  context.messagesEl.scrollTop=2380;
  context.renderSessionProgress('s');assert.equal(calls[1].preserveScroll,false);
});

test('sidebar redraw restores focus to the menu action or its trigger',()=>{
  const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
  const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
  const source=html.slice(html.indexOf('function renderSidebarLists('),html.indexOf('function projectSessionCountLabel('));
  const targets=[],context={state:{openSessionMenuId:'s'},CSS:{escape:x=>x},renderProjectList(){},renderChatList(){},positionSessionMenu(){},
    document:{activeElement:{dataset:{menuToggleId:'s'}},querySelector:selector=>({focus:()=>targets.push(selector)})}};
  vm.createContext(context);vm.runInContext(source,context);
  context.renderSidebarLists();assert.equal(targets[0],'[data-menu-id="s"]');
  context.document.activeElement.dataset={menuId:'s',menuAction:'move'};context.state.openSessionMenuId=null;
  context.renderSidebarLists();assert.equal(targets[1],'[data-menu-toggle-id="s"]');
});
