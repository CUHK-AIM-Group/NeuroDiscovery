const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../discovery-study.js'),'utf8');
const code=source.slice(source.indexOf('function significanceHTML('),source.indexOf('function feedbackGlossHTML('));
const pack=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v9.json'),'utf8'));
test('all 34 cases render their own frozen significance in both languages',()=>{
  const context={data:{cards:pack.cards,meta:pack.public_meta,session:{pack_id:pack.pack_id}},config:{pack_id:'another-pack'},significance:{},display:{language:'zh'},h:s=>s};
  vm.createContext(context);vm.runInContext(code,context);
  for(const card of pack.cards) for(const lang of ['zh','en']) {
    context.display.language=lang;
    assert(context.significanceHTML(card.id).includes(card.pre.significance[lang]));
  }
});
test('a historical session cannot inherit a newer pack significance sidecar',()=>{
  const context={data:{cards:[{id:'packet-01',pre:{}}],meta:{version_label:'v8'},session:{pack_id:'old'}},config:{pack_id:'new'},significance:{'packet-01':{zh:'new prose',en:'new prose'}},display:{language:'zh'},h:s=>s};
  vm.createContext(context);vm.runInContext(code,context);
  assert.equal(context.significanceHTML('packet-01'),'');
  context.config.pack_id='old';
  assert(context.significanceHTML('packet-01').includes('new prose'));
});
