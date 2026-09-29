// Package byte-identical production assets; never maintain a second demo UI.
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const source=path.resolve(__dirname,'../../core/web/static');
const destination=path.resolve(__dirname,'../demo/native-ui');
const files=[];
function copy(dir,relative=''){
  for(const entry of fs.readdirSync(dir,{withFileTypes:true})){
    if(entry.name==='tests'||entry.name==='__pycache__')continue;
    const name=path.join(relative,entry.name),input=path.join(source,name),output=path.join(destination,name);
    if(entry.isDirectory()){copy(input,name);continue;}
    if(!entry.isFile())continue;
    fs.mkdirSync(path.dirname(output),{recursive:true});fs.copyFileSync(input,output);
    files.push({path:name.replaceAll('\\','/'),sha256:crypto.createHash('sha256').update(fs.readFileSync(input)).digest('hex')});
  }
}
copy(source);
fs.writeFileSync(path.resolve(__dirname,'../demo/NATIVE_UI.json'),JSON.stringify({source:'core/web/static',byte_identical:true,files},null,2));
console.log(`Copied ${files.length} production frontend assets without modification.`);
