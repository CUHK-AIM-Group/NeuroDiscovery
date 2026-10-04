const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
for (const [stored,expected] of [[null,1.2],[1.1,1.2],[1,1.2],[1.3,1.3],[0.9,0.9]]) {
  test(`typography default migration preserves custom scale ${stored}`, () => {
    const data = new Map(stored === null ? [] : [['settings',JSON.stringify({textScale:stored})]]);
    const context = {state:{textScale:stored ?? 1.2,localSettings:{}},DEFAULT_LOCAL_SETTINGS:{textScale:1.2},
      SETTINGS_KEY:'settings',TEXT_SCALE_DEFAULT_MIGRATION_KEY:'migration',
      TEXT_SCALE_120_MIGRATION_KEY:'migration120',TEXT_SCALE_DEFAULT:1.2,
      clampTextScale:value=>value,localStorage:{getItem:key=>data.get(key),setItem:(key,value)=>data.set(key,value)}};
    vm.createContext(context);
    vm.runInContext(html.slice(html.indexOf('function loadLocalSettings('),html.indexOf('function rendererProviderNeedsNoApiKey(')),context);
    context.loadLocalSettings();
    assert.equal(context.state.textScale,expected);
    context.state.localSettings.textScale=1;
    context.saveLocalSettings();context.loadLocalSettings();
    assert.equal(context.state.textScale,1);
  });
}
