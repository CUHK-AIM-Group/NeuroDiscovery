const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../discovery-study.js'),'utf8');
const pack=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v10.json'),'utf8'));
const code=source.slice(source.indexOf('function significanceHTML('),source.indexOf('function feedbackGlossHTML('))
  +source.slice(source.indexOf('function partHTML('),source.indexOf('function hypothesisHTML('))
  +source.slice(source.indexOf('function probabilityText('),source.indexOf('function multiTopicMaterialHTML('))
  +source.slice(source.indexOf('function readableResultTablesHTML('),source.indexOf('function renderMaterial('));
const makeContext=()=>{
  const context={data:{cards:pack.cards,meta:pack.public_meta},config:{},significance:{},display:{language:'zh'},current:0,h:s=>String(s??''),safeUrl:x=>x,f:(x,d=3)=>typeof x==='number'?x.toFixed(d):'—',pc:x=>typeof x==='number'?`${(100*x).toFixed(2)}%`:'—'};
  vm.createContext(context);vm.runInContext(code,context);return context;
};
const analysisPack=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v14.json'),'utf8'));
const analyses=output=>[...output.matchAll(/<p class="result-analysis" data-user-content>(.*?)<\/p>/gs)].map(match=>match[1]);
test('every current and historical results table has one immediate bilingual analysis without changing materials',()=>{
  const context=makeContext();
  for(const selected of [pack,analysisPack]) {
    const before=JSON.stringify(selected);
    for(const card of selected.cards) for(const language of ['zh','en']) {
      context.display.language=language;
      const output=context.readableResultTablesHTML(card);
      const count=(output.match(/<table>/g)||[]).length;
      assert.equal(analyses(output).length,count,card.id);
      assert.equal((output.match(/<\/table><\/div><p class="result-analysis"/g)||[]).length,count,card.id);
      for(const text of analyses(output)) {
        assert(text.length>100,card.id);
        if(language==='en') assert(!/[\u4e00-\u9fff]/u.test(text),card.id);
      }
      if(!card.post.experimental_results) assert.match(output,/<p class="result-analysis"[^>]*>[^]*?<\/p><\/details>/);
      assert.equal(context.readableResultTablesHTML({...card,post:null}),'');
    }
    assert.equal(JSON.stringify(selected),before);
  }
});
test('analyses interpret cohort discrepancies, corrected results and sparse events',()=>{
  const context=makeContext();
  const render=id=>analyses(context.readableResultTablesHTML(analysisPack.cards.find(card=>card.id===id)));
  for(const language of ['zh','en']) {
    context.display.language=language;
    const genetics=render('packet-38');
    assert(genetics[0].includes('-0.034')&&genetics[0].includes('0.7579'));
    assert(genetics[1].includes('-0.199')&&genetics[1].includes('0/3'));
    assert(genetics[1].includes(language==='zh'?'证据集中在外部队列':'concentrated in the external cohort'));
    for(const id of ['packet-44','packet-46']) {
      const external=render(id)[1];
      assert(external.includes(language==='zh'?'仅有 7 个事件':'Only 7 events'));
      assert(external.includes(language==='zh'?'不能单凭这一点':'does not by itself'));
    }
    const temporal=render('packet-44')[1];
    assert(temporal.includes('0.0834'));
    assert(temporal.includes(language==='zh'?'区间不含 1，校正后 q 未达到':'interval excludes 1, q does not meet'));
    const connectivity=render('packet-01');
    assert(connectivity[0].includes('-0.552')&&connectivity[0].includes('-0.619'));
    assert(connectivity[1].includes('-0.594')&&connectivity[1].includes('99.40%'));
    assert(connectivity[2].includes('-0.582')&&connectivity[2].includes('-0.755'));
  }
});
test('analysis handles mixed signs, missing estimates, failed models and escaping',()=>{
  const context=makeContext();context.display.language='en';
  context.h=value=>String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  const rows=[{domain:'<img src=x>',standardized_beta:-0.2},{domain:'test',standardized_beta:0.4},{domain:'missing',standardized_beta:null}];
  const text=context.connectivityAnalysisHTML(rows,'external');
  assert(text.includes('do not share'));
  assert(text.includes('1 comparisons lack'));
  assert(text.includes('&lt;img src=x&gt;')&&!text.includes('<img'));
  const card=structuredClone(analysisPack.cards.find(card=>card.id==='packet-38'));
  card.post.experimental_results[0].internal.primary.effect=NaN;
  card.post.experimental_results[1].internal.failure='failed';
  card.post.experimental_results[2].internal=null;
  const missing=analyses(context.readableResultTablesHTML(card))[0];
  assert(missing.includes('No valid effect estimates'));
  assert(missing.includes('3 rows have missing'));
  assert(!missing.includes('NaN'));
  const prognosis=structuredClone(analysisPack.cards.find(card=>card.id==='packet-44'));
  prognosis.post.experimental_results[0].external.primary.hazard_ratio_ci_low=null;
  prognosis.post.experimental_results[0].external.q_value=null;
  const incomplete=analyses(context.readableResultTablesHTML(prognosis))[1];
  assert(incomplete.includes('the interval is unavailable, q is unavailable'));
});
test('legacy result renderers also append analyses immediately after their tables',()=>{
  const context=makeContext();
  context.domainName=value=>value;
  context.comparisonName=row=>row.domain;
  context.variantName=value=>value;
  vm.runInContext(source.slice(source.indexOf('function internalHTML('),source.indexOf('function feedbackHTML(')),context);
  const post=analysisPack.cards.find(card=>card.id==='packet-01').post;
  for(const language of ['zh','en']) {
    context.display.language=language;
    for(const output of [context.internalHTML(post.internal_rows),context.externalHTML(post.external_rows),context.externalHTML(post.external_rows,true)]) {
      assert.equal(analyses(output).length,1);
      assert.match(output,/<\/table><\/div><p class="result-analysis"/);
    }
  }
});
test('every new case shows its four readable sections in both languages',()=>{
  const context=makeContext();
  for(const card of pack.cards) for(const lang of ['zh','en']){
    context.display.language=lang;
    const output=context.readableMaterialHTML(card),reading=card.pre.reading;
    for(const key of ['rationale','methods','results'])assert(output.includes(reading[key][lang]),`${card.id} ${key}`);
    assert(output.includes(card.pre.significance[lang]));
    assert(output.includes(`<span>${card.pre.topic}</span>`),'Keep the topic as a standalone translation unit.');
    assert.equal((output.match(/class="material-part"/g)||[]).length,4);
    assert(output.includes('<table'),'Completed experiment values must remain visible.');
    assert(!/程序原始|精确定义|source_note|bootstrap_valid|OLS|残差 RMS|seed/.test(output));
    assert.equal(output.includes(lang==='en'?'Learning from earlier experiments':'前序实验的启发'),Boolean(reading.feedback));
    if(reading.feedback)assert(output.includes(reading.feedback[lang]));
    for(const ref of card.pre.references){assert(output.includes(ref.url));assert(output.includes(reading.references[ref.id][lang]));}
  }
});
test('every current case includes a bilingual plain-language terminology guide',()=>{
  const selected=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v14.json'),'utf8'));
  const context=makeContext();context.data={cards:selected.cards,meta:selected.public_meta};
  for(const card of selected.cards)for(const lang of ['zh','en']){
    context.display.language=lang;
    const output=context.readableMaterialHTML(card);
    assert(output.includes('class="terminology" data-user-content'),`${card.id} ${lang}: glossary present`);
    assert(!output.includes('class="terminology" open'),`${card.id} ${lang}: glossary collapsed by default`);
    assert(output.includes(lang==='zh'?'术语速查':'Quick terminology guide'),`${card.id} ${lang}: localized heading`);
    assert((output.match(/class="term-entry"/g)||[]).length>=8,`${card.id} ${lang}: enough explanations`);
    assert(output.includes(lang==='zh'?'不改变研究中的原始定义和实验结果':"do not change the study's original definitions or results"));
  }
  context.display.language='zh';
  assert(context.readableMaterialHTML(selected.cards.find(card=>card.id==='packet-01')).includes('功能连接'));
  assert(context.readableMaterialHTML(selected.cards.find(card=>card.id==='packet-36')).includes('等位基因剂量'));
  assert(context.readableMaterialHTML(selected.cards.find(card=>card.id==='packet-42')).includes('风险比（HR）'));
});
test('out-of-field readers get an explanation for every cohort, statistic and network noun',()=>{
  const selected=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v14.json'),'utf8'));
  const context=makeContext();context.data={cards:selected.cards,meta:selected.public_meta};
  const connectivity=selected.cards.find(card=>card.id==='packet-01');
  const genetics=selected.cards.find(card=>card.id==='packet-36');
  const prognosis=selected.cards.find(card=>card.id==='packet-46');
  context.display.language='zh';
  const connectivityText=context.readableMaterialHTML(connectivity);
  for(const label of ['四个队列（TCP / UCLA / COBRE / HCP-EP）','校正（协变量）','其他脑网络','双相障碍','P 值'])
    assert(connectivityText.includes(label),`connectivity case explains ${label}`);
  const geneticsText=context.readableMaterialHTML(genetics);
  for(const label of ['β（标准化回归系数）','脑区体积（结构性 MRI 指标）','痴呆（进展结局）','APOE ε4'])
    assert(geneticsText.includes(label),`imaging-genetics case explains ${label}`);
  const prognosisText=context.readableMaterialHTML(prognosis);
  for(const label of ['标准差（SD）','n / 事件数','Cox 生存分析','95% 置信区间（CI）'])
    assert(prognosisText.includes(label),`prognosis case explains ${label}`);
  // A region only used in the prognosis panel must not be pulled into a case that never mentions it.
  assert(!context.readableMaterialHTML(selected.cards.find(card=>card.id==='packet-37')).includes('痴呆（进展结局）'));
  context.display.language='en';
  for(const card of selected.cards){
    const output=context.readableMaterialHTML(card);
    assert(!/[\u3400-\u9fff]/.test(output.match(/<details class="terminology"[\s\S]*?<\/details>/)[0]),`${card.id}: translated glossary`);
  }
});
test('unrevealed outcomes are not displayed and old materials retain their branch',()=>{
  const context=makeContext(),card=structuredClone(pack.cards[0]);delete card.post;
  const output=context.readableMaterialHTML(card);
  assert(!output.includes(card.pre.reading.results.zh));
  assert(!output.includes('<table'));
  assert(source.includes('if(card.pre.reading) return readableMaterialHTML(card);'));
});
test('all 34 cards render their own internal and external measurements in both languages',()=>{
  const context=makeContext(),before=JSON.stringify(pack);
  for(const card of pack.cards)for(const lang of ['zh','en']){
    context.display.language=lang;
    const output=context.readableResultTablesHTML(card),post=card.post;
    if(lang==='en')assert(!/[\u3400-\u9fff]/.test(output),`${card.id}: translated table labels`);
    if(post.experimental_results){
      assert(output.includes('ADNI1/GO/2')&&output.includes('ADNI3'));
      for(const cfg of post.experimental_results)for(const phase of ['internal','external']){
        const row=cfg[phase];assert.equal(row.status,'complete');assert.equal(row.failure,null);
        assert(output.includes(context.f(post.result_kind==='imaging_genetics'?row.primary.effect:row.primary.hazard_ratio)),card.id);
        assert(output.includes(context.probabilityText(row.q_value)),card.id);
      }
    }else{
      assert(output.includes('TCP')&&output.includes('UCLA'));
      for(const row of [...post.internal_rows,...post.external_rows])assert(output.includes(context.f(row.standardized_beta)),card.id);
      for(const row of post.external_rows)assert(output.includes(context.pc(row.bootstrap_standardized_beta.bootstrap_direction_fraction)),card.id);
    }
  }
  assert.equal(JSON.stringify(pack),before,'Never alter results or cases while rendering.');
});
test('small probabilities are not displayed as zero and missing values remain missing',()=>{
  const context=makeContext();
  assert.equal(context.probabilityText(0.0000123),'1.23e-5');
  assert.equal(context.probabilityText(0.01445468375),'0.0145');
  assert.equal(context.probabilityText(null),'—');
  assert.equal(context.probabilityText(NaN),'—');
});
test('the success-led ten-case panel retains every result table and bilingual narrative',()=>{
  const selected=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v11.json'),'utf8'));
  const context=makeContext();context.data={cards:selected.cards,meta:selected.public_meta};
  assert.equal(selected.cards.length,10);
  assert.equal(selected.cards.filter(card=>card.pre.topic==='跨诊断脑连接').length,4);
  for(const card of selected.cards)for(const lang of ['zh','en']){
    context.display.language=lang;
    const output=context.readableMaterialHTML(card);
    assert(output.includes(card.pre.reading.results[lang]),card.id);
    assert(output.includes(card.pre.significance[lang]),card.id);
    assert(output.includes('<table'),card.id);
    assert.equal((output.match(/class="material-part"/g)||[]).length,4);
    assert(!output.includes('main_supported')&&!output.includes('supplementary_partial'));
  }
});

test('restored context displays every complete citation and its specific relevance in both languages',()=>{
  const revised=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v12.json'),'utf8'));
  const context=makeContext();context.data={cards:revised.cards,meta:revised.public_meta};
  const before=JSON.stringify(revised);
  for(const card of revised.cards)for(const lang of ['zh','en']){
    context.display.language=lang;
    const output=context.readableMaterialHTML(card),reading=card.pre.reading;
    assert(output.includes(reading.scale[lang]),card.id);
    assert(output.includes(reading.methods[lang])&&output.includes(reading.results[lang]),card.id);
    assert(output.includes(`${lang==='en'?'Related studies':'相关研究'} (${card.pre.references.length})`));
    assert.equal((output.match(/class="related-study"/g)||[]).length,card.pre.references.length);
    for(const ref of card.pre.references){
      const detail=reading.reference_details[ref.id];
      for(const value of [ref.url,ref.pmid,detail.title,detail.did[lang],detail.relation[lang]])assert(output.includes(value),`${card.id} ${ref.id}`);
    }
    const tables=context.readableResultTablesHTML(card);
    if(card.post.internal_rows)for(const row of [...card.post.internal_rows,...card.post.external_rows])assert(tables.includes(`${row.cases} / ${row.controls}`),card.id);
    else for(const cfg of card.post.experimental_results)for(const phase of ['internal','external'])assert(tables.includes(String(cfg[phase].primary.n)));
  }
  assert.equal(JSON.stringify(revised),before);
});

test('reference prose and verified titles are escaped rather than interpreted as HTML',()=>{
  const revised=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v12.json'),'utf8'));
  const context=makeContext(),card=structuredClone(revised.cards[0]);
  context.h=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  card.pre.reading.reference_details.P1.did.zh='<img src=x onerror=alert(1)>';
  const output=context.readableMaterialHTML(card);
  assert(!output.includes('<img'));
  assert(output.includes('&lt;img'));
});

test('forward research appears only for documented cases after their full results',()=>{
  const revised=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v13.json'),'utf8'));
  const context=makeContext();context.data={cards:revised.cards,meta:revised.public_meta};
  const before=JSON.stringify(revised);
  for(const card of revised.cards)for(const lang of ['zh','en']){
    context.display.language=lang;
    const output=context.readableMaterialHTML(card),note=card.pre.reading.next_research;
    assert.equal(output.includes('data-next-research'),Boolean(note),card.id);
    assert.equal((output.match(/class="material-part"/g)||[]).length,note?5:4);
    assert(output.includes(context.readableResultTablesHTML(card)),card.id);
    if(note){
      assert(output.includes(note[lang]));
      assert(output.includes(lang==='en'?'How this case informed the next research round':'本例结果如何进入下一轮研究'));
      assert(output.indexOf('data-next-research')>output.lastIndexOf('</table>'));
    }
    assert.equal(output.includes(lang==='en'?'Learning from earlier experiments':'前序实验的启发'),Boolean(card.pre.reading.feedback));
  }
  assert.equal(JSON.stringify(revised),before);
});

test('forward notes are hidden without results or complete bilingual text and remain escaped',()=>{
  const revised=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v13.json'),'utf8'));
  const context=makeContext(),card=structuredClone(revised.cards.find(c=>c.pre.reading.next_research));
  context.h=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  card.pre.reading.next_research.zh='<img src=x onerror=alert(1)>';
  let output=context.readableMaterialHTML(card);
  assert(output.includes('&lt;img')&&!output.includes('<img'));
  card.pre.reading.next_research.en=' ';
  assert(!context.readableMaterialHTML(card).includes('data-next-research'));
  card.pre.reading.next_research.en='A documented follow-up';delete card.post;
  assert(!context.readableMaterialHTML(card).includes('data-next-research'));
});

test('concrete parent and follow-up hypotheses retain full results and references in both languages',()=>{
  const revised=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v14.json'),'utf8'));
  const context=makeContext();context.data={cards:revised.cards,meta:revised.public_meta};
  const before=JSON.stringify(revised);
  for(const card of revised.cards)for(const lang of ['zh','en']){
    context.display.language=lang;
    const output=context.readableMaterialHTML(card),reading=card.pre.reading;
    const chains=[reading.feedback_chain,reading.next_research_chain].filter(Boolean);
    assert.equal(chains.length,1,card.id);
    assert.equal((output.match(/class="research-chain"/g)||[]).length,chains.length);
    for(const chain of chains){
      for(const value of [chain.source,chain.feedback,chain.change,...chain.next,...(chain.outcome?[chain.outcome]:[])])assert(output.includes(value[lang]),card.id);
      assert(output.includes(chain.mode==='explicit_parent'?'data-research-chain="parent-child"':'data-research-chain="round-followup"'));
      const chainHTML=context.researchChainHTML(chain,Boolean(reading.next_research_chain));
      if(lang==='en')assert(!/[\u3400-\u9fff]/.test(chainHTML));
      if(chain.mode==='round_followup')assert(!/父假设|子假设|Parent hypothesis|Child hypothesis/.test(chainHTML));
    }
    assert(output.includes(context.readableResultTablesHTML(card)));
    for(const key of ['rationale','scale','methods','results'])assert(output.includes(reading[key][lang]));
    for(const ref of card.pre.references)assert(output.includes(ref.url)&&output.includes(reading.reference_details[ref.id].did[lang]));
  }
  assert.equal(JSON.stringify(revised),before);
});

test('lineage distinguishes the present case and keeps separate later candidates readable',()=>{
  const revised=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v14.json'),'utf8'));
  const context=makeContext();
  const prior=revised.cards.find(c=>c.id==='packet-02').pre.reading.feedback_chain;
  const forward=revised.cards.find(c=>c.id==='packet-05').pre.reading.next_research_chain;
  assert(context.researchChainHTML(prior).includes('子假设 · 本例'));
  assert(context.researchChainHTML(forward,true).includes('父假设 · 本例'));
  const multiple=revised.cards.find(c=>c.id==='packet-37').pre.reading.next_research_chain;
  assert.equal((context.researchChainHTML(multiple,true).match(/<li>/g)||[]).length,2);
});

test('structured lineage text is escaped and future material stays hidden until outcomes are shown',()=>{
  const revised=JSON.parse(fs.readFileSync(path.join(__dirname,'../../study_materials/cs1_discovery_pilot_v14.json'),'utf8'));
  const context=makeContext(),card=structuredClone(revised.cards.find(c=>c.id==='packet-37'));
  context.h=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  card.pre.reading.next_research_chain.next[0].zh='<img src=x onerror=alert(1)>';
  const output=context.readableMaterialHTML(card);
  assert(output.includes('&lt;img')&&!output.includes('<img'));
  delete card.post;
  assert(!context.readableMaterialHTML(card).includes('research-chain'));
});
