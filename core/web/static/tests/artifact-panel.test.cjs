const {test}=require('node:test');
const assert=require('node:assert/strict');
const A=require('../artifact-panel.js');
const context={origin:'http://localhost:7080',chatId:'chat-1'};
test('artifact references stay local and tied to the current conversation',()=>{
  const file=A.fileReference('/demo/artifacts/run/hypotheses.csv',context);
  assert.equal(file.kind,'table');assert.equal(file.name,'hypotheses.csv');
  assert.equal(A.fileReference('reports/脑 影像.md',context).reference,'reports/脑 影像.md');
  assert.equal(A.fileReference('C:\\research\\report.pdf',context).kind,'pdf');
  assert.equal(A.fileReference('C:%5Cresearch%5Creport.pdf',context).reference,'C:\\research\\report.pdf');
  for(const value of ['javascript:alert(1)','data:text/html,bad','file:///C:/secrets.txt','https://example.com/paper.pdf','//server/file.csv','/api/env.json','#section'])assert.equal(A.fileReference(value,context),null,value);
  const other='/api/workbench/artifact?chat_id=chat-2&path=report.md';
  assert.equal(A.fileReference(other,context),null);
  assert.equal(A.fileReference(file.url.replace('/demo','http://localhost:7080/demo'),context).url,file.url);
});
test('CSV/TSV parsing preserves quoted commas, escaped quotes, multiline, BOM, blank cells, and formulas as text',()=>{
  const result=A.parseDelimited('\uFEFFid,text,score\r\nH1,"a, b\nwith ""quotes""",90\r\nH2,=SUM(A1:A2),\r\n');
  assert.deepEqual(result,{rows:[['id','text','score'],['H1','a, b\nwith "quotes"','90'],['H2','=SUM(A1:A2)','']],warning:false,truncated:false});
  assert.deepEqual(A.parseDelimited('x\ty\n1\t"two\tthree"','\t').rows,[['x','y'],['1','two\tthree']]);
  assert.deepEqual(A.parseDelimited('').rows,[]);
  assert.equal(A.parseDelimited('a\n"unfinished').warning,true);
  assert.equal(A.parseDelimited('a,b,c\n1,2,3',',',1,2).truncated,true);
  assert.equal(A.parseDelimited('a\n1\n2\n3',',',2).truncated,true);
});
test('known research formats route to an actual renderer; binaries use download fallback',()=>{
  for(const [name,kind] of [['REPORT.md','markdown'],['scores.TSV','table'],['config.json','json'],['model.py','text'],['figure.SVG','image'],['paper.pdf','pdf'],['plot.html','html'],['weights.pt','file']])assert.equal(A.fileKind(name),kind);
});
