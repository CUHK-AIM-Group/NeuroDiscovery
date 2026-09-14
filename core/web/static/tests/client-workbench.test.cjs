const test = require('node:test');
const assert = require('node:assert/strict');
const W = require('../client-workbench.js');

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
  const targets=[],context={state:{openSessionMenuId:'s'},CSS:{escape:x=>x},renderProjectList(){},renderChatList(){},
    document:{activeElement:{dataset:{menuToggleId:'s'}},querySelector:selector=>({focus:()=>targets.push(selector)})}};
  vm.createContext(context);vm.runInContext(source,context);
  context.renderSidebarLists();assert.equal(targets[0],'[data-menu-id="s"]');
  context.document.activeElement.dataset={menuId:'s',menuAction:'move'};context.state.openSessionMenuId=null;
  context.renderSidebarLists();assert.equal(targets[1],'[data-menu-toggle-id="s"]');
});
