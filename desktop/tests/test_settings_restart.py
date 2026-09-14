"""Offline settings restart policy and real save-handler regressions."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = r'''
const assert = require('node:assert/strict');
const {createRestartTracker} = require('./desktop/settings-restart');
const config = {host:'127.0.0.1',port:7082,runtimeMode:'python',pythonExe:'/fixture/python',
 localPythonExe:'',condaExe:'/fixture/conda',condaEnv:'nd',repoRoot:'/fixture/repo',
 environmentFile:'/fixture/env.json',fslDir:'',proxyUrl:'',language:'English',theme:'light',
 llmProvider:'openai',llmModel:'model-a',llmAddedModels:['model-a','model-b'],
 llmBaseUrl:'https://fixture.example/v1',llmApiKey:'synthetic',llmApiKeyEnv:'FIXTURE_KEY',
 llmApiKeyFile:'',llmApiKeySlot:'primary'};
'''


def node_test(program):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js required')
    result = subprocess.run([node, '-e', BASE + program], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_restart_only_for_effective_startup_settings():
    node_test(r'''
const tracker=createRestartTracker(config);
assert.equal(tracker.requiresRestart({...config}),false);
for(const [key,value] of Object.entries({language:'Simplified Chinese',theme:'dark',textScale:1.2,
 defaultShell:'cmd',fileOpenTarget:'Explorer',showAdvancedLogs:true,outputRoot:'/outputs',
 condaExe:'/inactive/conda',condaEnv:'inactive',llmApiKeySlot:'reserve'})) {
 assert.equal(tracker.requiresRestart({...config,[key]:value}),false,key);
}
for(const [key,value] of Object.entries({host:'localhost',port:7083,runtimeMode:'conda',
 pythonExe:'/other/python',localPythonExe:'/custom/python',repoRoot:'/other/repo',
 environmentFile:'/other/env.json',fslDir:'/fsl',proxyUrl:'http://127.0.0.1:7890',
 llmProvider:'anthropic',llmModel:'model-b',llmAddedModels:['model-a'],
 llmBaseUrl:'https://other.example/v1',llmApiKey:'other-synthetic',llmApiKeyEnv:'OTHER_KEY',
 llmApiKeyFile:'/fixture/keys.txt',llmApiMode:'responses',llmReasoningEffort:'high',
 llmThinkingMode:'enabled',llmMaxOutputTokens:'2048',llmTemperature:'0'})) {
 assert.equal(tracker.requiresRestart({...config,[key]:value}),true,key);
}
const conda={...config,runtimeMode:'conda'};
assert.equal(createRestartTracker(conda).requiresRestart({...conda,pythonExe:'/unused',localPythonExe:'/unused'}),false);
assert.equal(createRestartTracker(conda).requiresRestart({...conda,condaEnv:'new'}),true);
const file={...config,llmApiKeyFile:'/fixture/keys.txt'};
assert.equal(createRestartTracker(file).requiresRestart({...file,llmApiKeySlot:'reserve'}),true);
''')


def test_defaults_repeated_saves_and_reverts_do_not_create_false_prompts():
    node_test(r'''
const legacy={...config,llmAddedModels:null};
const tracker=createRestartTracker(legacy);
assert.equal(tracker.requiresRestart({...legacy,llmAddedModels:['model-a'],
 llmApiMode:'auto',llmReasoningEffort:'default',llmThinkingMode:'default',
 llmMaxOutputTokens:'',llmTemperature:''}),false);
assert.equal(createRestartTracker(config).requiresRestart({...config,llmAddedModels:['model-b','model-a']}),false);
assert.equal(createRestartTracker({...config,llmTemperature:0}).requiresRestart({...config,llmTemperature:'0.0'}),false);
assert.equal(tracker.requiresRestart({...legacy,port:7083}),true);
assert.equal(tracker.requiresRestart({...legacy,port:7083,language:'Simplified Chinese'}),true);
assert.equal(tracker.requiresRestart({...legacy,language:'Simplified Chinese'}),false);
const baseline={...config,llmAddedModels:['model-a','model-b']};
const immutable=createRestartTracker(baseline);
baseline.llmAddedModels.push('model-c');
assert.equal(immutable.requiresRestart(baseline),true);
''')


def test_desktop_get_and_save_handlers_keep_pending_changes_until_reverted():
    node_test(r'''
const fs=require('node:fs'), vm=require('node:vm');
const source=fs.readFileSync('./desktop/main.js','utf8');
const handlers={}; let saved={...config}; let menus=0;
const context={require, path:require('node:path'),app:{isPackaged:false,getPath:()=>'/fixture'},process:{platform:'test'},
 ipcMain:{handle:(name,handler)=>handlers[name]=handler},loadConfig:()=>({...saved}),
 saveConfig:next=>(saved={...saved,...next}),userConfigPath:()=>'/fixture/config.json',
 describeLlmConnectionStatus:()=>({}),setApplicationMenu:()=>menus++,
 settingsRestartTracker:createRestartTracker(config)};
vm.createContext(context);
for(const name of ['get-config','save-config']){
 const handler=source.match(new RegExp("^ipcMain.handle\\('neuroclaw:"+name+"'[\\s\\S]+?^\\}\\);",'m'));
 assert.ok(handler,name);vm.runInContext(handler[0],context);
}
const save=next=>handlers['neuroclaw:save-config'](null,{...saved,...next});
assert.equal(save({language:'Simplified Chinese'}).restartRequired,false);
assert.equal(menus,1);
assert.equal(save({port:7083}).restartRequired,true);
assert.equal(save({language:'English'}).restartRequired,true);
assert.equal(handlers['neuroclaw:get-config']().restartRequired,true);
assert.equal(save({port:7082}).restartRequired,false);
assert.equal(save({}).restartRequired,false);
assert.equal(handlers['neuroclaw:get-config']().restartRequired,false);
assert.ok(source.includes("createRestartTracker(launchConfig)"));
''')


def test_frontend_save_hides_button_and_omits_restart_copy_when_unnecessary():
    node_test(r'''
const fs=require('node:fs'),vm=require('node:vm');
require('./core/web/static/model-library');
const html=fs.readFileSync('./core/web/static/index.html','utf8');
let reply={config,restartRequired:false},status='';
const context={window:{NDModelLibrary:globalThis.NDModelLibrary,neuroclawDesktop:{saveConfig:async()=>reply}},
 state:{desktopConfig:{...config},localSettings:{},textScale:1,llmConnectionStatus:{apiKeyRequired:false,endpointConfigured:true}},
 settingsRestartBtn:{hidden:false},DEFAULT_DESKTOP_CONFIG:{},collectSettingsFromForm:()=>{},
 fetch:async()=>({ok:true,json:async()=>({})}),clampTextScale:value=>value,applyTextScale:()=>{},
 applyLanguage:()=>{},saveLocalSettings:()=>{},applyLlmConnectionStatus:()=>{},renderSettings:()=>{},
 uiText:(en,zh)=>en,setSettingsStatus:text=>status=text};
vm.createContext(context);
for(const name of ['applySettingsRestartStatus','saveSettings']) {
 const fn=html.match(new RegExp('^  (?:async )?function '+name+'\\([^\\n]*\\)[\\s\\S]+?^  }','m'));
 assert.ok(fn,name);vm.runInContext(fn[0],context);
}
(async()=>{
 await context.saveSettings();assert.equal(context.settingsRestartBtn.hidden,true);assert.equal(status,'Saved.');
 reply={config,restartRequired:true};await context.saveSettings();
 assert.equal(context.settingsRestartBtn.hidden,false);assert.ok(status.includes('Restart'));
 reply={config,restartRequired:false};await context.saveSettings();
 assert.equal(context.settingsRestartBtn.hidden,true);assert.equal(status,'Saved.');
 assert.ok(!html.includes('Model choices are a draft. Save and restart'));
 assert.ok(!html.includes('Saved. Interface changes apply now;'));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')
