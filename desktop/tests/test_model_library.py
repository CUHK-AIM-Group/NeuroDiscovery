"""User-selected model libraries: offline fixtures, no account requests."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def node_test(script):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required')
    result = subprocess.run([node, '-e', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_discovery_does_not_save_or_add_models_and_never_leaks_auth_errors():
    node_test(r'''
const assert = require('node:assert/strict');
const {selectedModelIds, discoverModels} = require('./desktop/model-library');
const cfg = {llmProvider:'openai', llmBaseUrl:'https://fixture.example/v1', llmApiKey:'synthetic-key', llmAddedModels:['chosen'], llmModel:'chosen'};
const original = JSON.stringify(cfg);
const requests = [];
(async () => {
 const result = await discoverModels(cfg, async (url, options) => {
   requests.push({url, options});
   return {ok:true, json:async () => ({data:[{id:'z-2'},{id:'z-1'},{id:'z-2'}, {id:null}]})};
 }, {});
 assert.deepEqual(result.models.map(m=>m.model), ['z-1','z-2']);
 assert.equal(JSON.stringify(cfg), original);
 assert.equal(requests[0].url, 'https://fixture.example/v1/models');
 assert.equal(requests[0].options.headers.Authorization, 'Bearer synthetic-key');
 assert.equal(requests[0].options.redirect, 'error');
 assert.deepEqual(selectedModelIds(cfg), ['chosen']);
 assert.deepEqual(selectedModelIds({llmModel:'legacy'}), ['legacy']);
 assert.deepEqual(selectedModelIds({llmModel:'legacy',llmAddedModels:[]}), []);
 assert.deepEqual(selectedModelIds({llmAddedModels:[' a ','a','b']}), ['a','b']);
 for (const value of ['not-an-array', [null], [''], ['bad\nid'], Array(501).fill('a')]) assert.throws(()=>selectedModelIds({llmAddedModels:value}));
 let calls=0;
 for (const status of [401,403,404,429,500]) {
   await assert.rejects(discoverModels(cfg, async()=>{calls++;return {ok:false,status,text:async()=>cfg.llmApiKey};},{}), error => !error.message.includes(cfg.llmApiKey) && error.message.includes(String(status)));
 }
 assert.equal(calls,5, 'Never retry or switch keys');
 await assert.rejects(discoverModels(cfg, async()=>{throw new Error(cfg.llmApiKey)},{}),error=>!error.message.includes(cfg.llmApiKey));
 await assert.rejects(discoverModels(cfg, async()=>({ok:true,json:async()=>{throw new Error(cfg.llmApiKey)}}),{}), /valid JSON/);
 for (const llmBaseUrl of ['https://user:password@fixture.example/v1','file:///keys.txt','https://fixture.example/v1?api_key=secret','http://remote.example/v1']) {
   await assert.rejects(discoverModels({...cfg,llmBaseUrl}, async()=>{assert.fail('must reject before request')},{}));
 }
 await assert.rejects(discoverModels({...cfg,llmProvider:'ollama_cloud'},async()=>assert.fail('must not send an Ollama key to a proxy'),{}), /Ollama Cloud requires/);
 await assert.rejects(discoverModels(cfg,async()=>({ok:true,json:async()=>null}),{}), /compatible model list/);
 const paged=[];
 const claude=await discoverModels({...cfg,llmProvider:'anthropic',llmBaseUrl:'https://fixture.example'},async(url, options)=>{
   paged.push(url); assert.equal(options.headers['x-api-key'],'synthetic-key'); assert.ok(!options.headers.Authorization);
   return {ok:true,json:async()=>paged.length===1?{data:[{id:'claude-one'}],has_more:true,last_id:'one'}:{data:[{id:'claude-two'}],has_more:false}};
 },{});
 assert.deepEqual(paged,['https://fixture.example/v1/models','https://fixture.example/v1/models?after_id=one']);
 assert.equal(claude.models.length,2);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_frontend_requires_explicit_selection_and_ignores_stale_discovery():
    node_test(r'''
const assert = require('node:assert/strict');
require('./core/web/static/model-library');
const {ids,create}=globalThis.NDModelLibrary;
let cfg={llmProvider:'openai',llmBaseUrl:'https://fixture.example/v1',llmModel:'saved',llmAddedModels:['saved']};
let resolve,notify=0;
const library=create({getConfig:()=>cfg, discover:()=>new Promise(r=>resolve=r),changed:()=>notify++});
(async()=>{
 const first=library.fetch();
 resolve({ok:true,models:[{model:'one'},{model:'two'},{model:'three'}]}); await first;
 assert.deepEqual(library.state.selected,[]);
 assert.deepEqual(ids(cfg),['saved']);
 library.toggle('one',true); library.toggle('three',true); library.toggle('unfetched',true);
 library.addSelected(); assert.deepEqual(ids(cfg),['saved','one','three']);
 assert.equal(cfg.llmModel,'saved');
 library.makeDefault('three');assert.equal(cfg.llmModel,'three');
 library.remove('three'); assert.equal(cfg.llmModel,'saved');
 library.remove('saved'); library.remove('one'); assert.deepEqual(ids(cfg),[]); assert.equal(cfg.llmModel,'');
 assert.equal(library.addManual(''),false);
 assert.equal(library.addManual('bad\nid'),false);
 assert.equal(library.addManual('manual-id'),true); assert.deepEqual(ids(cfg),['manual-id']);
 const stale=library.fetch();
 cfg={...cfg,llmBaseUrl:'https://new.example/v1',llmAddedModels:[],llmModel:''};
 library.invalidate(); resolve({ok:true,models:[{model:'stale-model'}]}); await stale;
 assert.deepEqual(library.state.models,[]); assert.deepEqual(ids(cfg),[]);
 const failed=library.fetch();resolve({ok:false,message:'HTTP 401'});await failed;
 assert.equal(library.state.status,'error'); assert.deepEqual(ids(cfg),[]);
 assert.ok(notify>0);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_bootstrap_and_chat_picker_never_discover_models():
    html = (ROOT / 'core/web/static/index.html').read_text(encoding='utf-8')
    assert 'refreshAvailableModels' not in html
    assert "fetch('/api/env/models'" not in html
    assert 'data-manage-models' in html
    assert 'data-model-action="fetch"' in html
    assert 'data-model-action="add"' in html
    assert 'data-default-model=' in html
    assert "state.desktopConfig.llmAddedModels = []" in html
    assert "state.desktopConfig.llmApiKeyFile = ''" in html
