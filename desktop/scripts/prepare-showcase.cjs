/* Reproducible synthetic fixtures, built with Node only. No downloads or credentials. */
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const D=require('../demo/scenarios.js');
const desktop=path.resolve(__dirname,'..');
const output=path.resolve(desktop,'../data/demo');
const digest=data=>crypto.createHash('sha256').update(data).digest('hex');
const records=[];

function write(filename,data,shape) {
  const target=path.join(output,filename);
  const bytes=Buffer.isBuffer(data)?data:Buffer.from(data,'utf8');
  if(fs.existsSync(target)&&!fs.readFileSync(target).equals(bytes))throw new Error('Preserving changed file; choose a fresh fixture version: '+target);
  fs.mkdirSync(path.dirname(target),{recursive:true});
  if(!fs.existsSync(target))fs.writeFileSync(target,bytes,{flag:'wx'});
  records.push({path:filename.replaceAll('\\','/'),bytes:bytes.length,sha256:digest(bytes),...(shape?{shape}:{})});
}
function npy(values,shape,dtype='<f4') {
  if(shape.reduce((a,b)=>a*b,1)!==values.length)throw new Error('NPY shape mismatch');
  let header=`{'descr': '${dtype}', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length===1?',':''}), }`;
  header+=' '.repeat((64-(10+header.length+1)%64)%64)+'\n';
  const width=dtype==='<i8'?8:4, bytes=Buffer.alloc(10+header.length+values.length*width);
  bytes.set([0x93,0x4e,0x55,0x4d,0x50,0x59,1,0]);bytes.writeUInt16LE(header.length,8);bytes.write(header,10,'ascii');
  for(let i=0;i<values.length;i++) {
    const v=Number(values[i]);if(!Number.isFinite(v))throw new Error('Non-finite fixture');
    if(dtype==='<i8')bytes.writeBigInt64LE(BigInt(v),10+header.length+i*width);else bytes.writeFloatLE(v,10+header.length+i*width);
  }
  return bytes;
}
let seed=20260929;
function random(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return(seed+1)/4294967297;}
function normal(){return Math.sqrt(-2*Math.log(random()))*Math.cos(2*Math.PI*random());}
const splitFor=i=>i<8?'train':i<10?'validation':'test';
const siteFor=i=>i<5?'A':i<10?'B':'C';
const subject=i=>'demo-'+String(i+1).padStart(3,'0');
function json(filename,data){write(filename,JSON.stringify(data,null,2)+'\n');}
function splitFiles(id,windows=1) {
  const splits={train:[],validation:[],test:[]}, rows=['sample_index,subject_id,site,label,split'];
  for(let i=0;i<12*windows;i++) {
    const person=Math.floor(i/windows), split=splitFor(person);
    splits[split].push(i);rows.push([i,subject(person),siteFor(person),windows===1?i%2:i%4,split].join(','));
  }
  json(id+'/model_ready/splits.json',splits);
  write(id+'/model_ready/subjects.csv',rows.join('\n')+'\n');
  write(id+'/source/phenotype.csv',rows.join('\n')+'\n');
  return splits;
}
function mri(id) {
  const n=12,t=80,r=12;
  const times=new Float32Array(n*t*r),volumes=new Float32Array(n*r*r*r),readyT1=new Float32Array(volumes.length),fc=new Float32Array(n*r*r);
  for(let i=0;i<times.length;i++)times[i]=normal();
  for(let s=0;s<n;s++) {
    for(let roi=0;roi<r;roi++) {
      let mean=0;for(let a=0;a<t;a++)mean+=times[(s*t+a)*r+roi]/t;
      for(let a=0;a<t;a++)times[(s*t+a)*r+roi]-=mean;
    }
    for(let a=0;a<r;a++)for(let b=0;b<r;b++) {
      let cross=0,aa=0,bb=0;
      for(let tt=0;tt<t;tt++){const x=times[(s*t+tt)*r+a],y=times[(s*t+tt)*r+b];cross+=x*y;aa+=x*x;bb+=y*y;}
      const corr=Math.max(-.99999,Math.min(.99999,cross/Math.sqrt(aa*bb)));
      fc[(s*r+a)*r+b]=a===b?0:Math.atanh(corr);
    }
    const length=r*r*r;let sum=0,sum2=0;
    for(let v=0;v<length;v++){const value=100+20*normal();volumes[s*length+v]=value;sum+=value;sum2+=value*value;}
    const mean=sum/length,sd=Math.sqrt(sum2/length-mean*mean);
    for(let v=0;v<length;v++)readyT1[s*length+v]=(volumes[s*length+v]-mean)/sd;
  }
  write(id+'/source/roi_timeseries.npy',npy(times,[n,t,r]),[n,t,r]);
  write(id+'/source/t1.npy',npy(volumes,[n,r,r,r]),[n,r,r,r]);
  write(id+'/model_ready/fc_z.npy',npy(fc,[n,r,r]),[n,r,r]);
  write(id+'/model_ready/t1.npy',npy(readyT1,[n,1,r,r,r]),[n,1,r,r,r]);
  write(id+'/model_ready/labels.npy',npy(Array.from({length:n},(_,i)=>i%2),[n],'<i8'),[n]);
  splitFiles(id);
  json(id+'/model_ready/qc.json',{synthetic:true,atlas:'demo12',real_spatial_preprocessing:false,subjects:n,finite:true,fc_symmetric:true,fc_diagonal:0,subject_overlap:0,normalization:'T1 within each sample; FC computed within each sample',not_for_scientific_inference:true});
}
function eeg() {
  const n=24,c=62,t=800,raw=new Float32Array(n*c*t),ready=new Float32Array(n*c*t);
  for(let i=0;i<raw.length;i++)raw[i]=normal()*8+Math.sin(i%t*2*Math.PI*10/200);
  const means=[],stds=[];
  for(let channel=0;channel<c;channel++) {
    let sum=0,sum2=0;
    for(let s=0;s<16;s++)for(let k=0;k<t;k++){const v=raw[(s*c+channel)*t+k];sum+=v;sum2+=v*v;}
    const mean=sum/(16*t),sd=Math.sqrt(sum2/(16*t)-mean*mean);means.push(mean);stds.push(sd);
    for(let s=0;s<n;s++)for(let k=0;k<t;k++){const i=(s*c+channel)*t+k;ready[i]=(raw[i]-mean)/sd;}
  }
  write('seediv/source/epochs.npy',npy(raw,[n,c,t]),[n,c,t]);
  write('seediv/model_ready/epochs.npy',npy(ready,[n,c,t]),[n,c,t]);
  write('seediv/model_ready/labels.npy',npy(Array.from({length:n},(_,i)=>i%4),[n],'<i8'),[n]);
  splitFiles('seediv',2);
  json('seediv/model_ready/qc.json',{synthetic:true,subjects:12,windows:24,sampling_hz:200,window_seconds:4,finite:true,subject_overlap:0,normalization_fit_sample_indices:Array.from({length:16},(_,i)=>i),channel_mean:means,channel_std:stds,filtering_executed:false,not_for_scientific_inference:true});
}
for(const id of ['adhd200','abide','adni'])mri(id);
eeg();
write('load_demo.py',fs.readFileSync(path.join(desktop,'demo/load_demo.py')));
write('README.md','# NeuroDiscovery synthetic demo data\n\nNo real participants or downloaded research data. These are generated interface fixtures, not miniature real cohorts. All IDs, labels, sites and values are synthetic.\n\n12 subjects per cohort; MRI uses toy demo12 ROIs and 12³ voxels. SEED-IV-shaped EEG has two 4-second windows per synthetic subject at 200 Hz. No real spatial registration, segmentation, artifact removal or atlas extraction was executed.\n\nNode builder actually computes FC from synthetic time series and normalizes EEG with training-subject statistics only. Source arrays are retained. Train / validation / test = 8 / 2 / 2 subjects. Site C is the toy test site.\n\nUse load_demo.py with NumPy; --torch loads tensors; --export-pt converts MRI records into the project input dictionary using PyTorch. Clinical validation needs real data, sufficient samples, external replication and a preregistered protocol. UI experiment scores are scripted, not measured from these files.\n');
json('MANIFEST.json',{schema_version:1,generator:'desktop/scripts/prepare-showcase.cjs',synthetic:true,downloaded:false,scientific_validation:false,seed:20260929,files:records.slice()});
const ideas=path.join(desktop,'demo-workspace/ideas/adhd_network');fs.mkdirSync(ideas,{recursive:true});
for(const [filename,content]of Object.entries(D.artifacts('idea'))) {
  const target=path.join(ideas,filename);
  if(fs.existsSync(target)&&fs.readFileSync(target,'utf8')!==content)throw new Error('Preserving changed idea: '+target);
  if(!fs.existsSync(target))fs.writeFileSync(target,content,{flag:'wx'});
}
const theme=fs.readFileSync(path.resolve(desktop,'../core/web/static/workspace-tokens.css'));
fs.writeFileSync(path.join(desktop,'demo/theme.css'),theme);
console.log(JSON.stringify({ok:true,path:output,synthetic:true,files:records.length,bytes:records.reduce((s,r)=>s+r.bytes,0)}));
