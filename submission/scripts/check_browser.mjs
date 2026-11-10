import fs from 'node:fs/promises';
import {predict,transform} from '../website/engine.mjs';
const root=new URL('../',import.meta.url);
const fixtures=JSON.parse(await fs.readFile(new URL('.build/browser_fixtures.json',root),'utf8'));
const checks=[];
const preprocessing=JSON.parse(await fs.readFile(new URL('.build/preprocessing_fixtures.json',root),'utf8'));
for(const f of preprocessing){
 const c=JSON.parse(await fs.readFile(new URL(`website/dist/models/${f.scenario}_contract.json`,root),'utf8'));
 let max=0;
 f.raw.forEach((raw,i)=>{const x=transform(raw,c);x.forEach((v,j)=>max=Math.max(max,Math.abs(v-f.expected[i][j])));});
 checks.push({scenario:f.scenario,model:'preprocessing',rows:f.raw.length,max_absolute_error:max,label_mismatches:0});
}
for(const f of fixtures){
 const model=JSON.parse(await fs.readFile(new URL(`website/dist/models/${f.scenario}_${f.model}.json`,root),'utf8'));
 let max=0,mismatch=0;
 f.x.forEach((x,i)=>{const p=predict(model,x);max=Math.max(max,Math.abs(p-f.expected[i]));if((p>=model.threshold)!==(f.expected[i]>=model.threshold))mismatch++;});
 checks.push({scenario:f.scenario,model:f.model,rows:f.x.length,max_absolute_error:max,label_mismatches:mismatch});
}
await fs.writeFile(new URL('results/browser_parity.json',root),JSON.stringify(checks,null,2));
console.log(checks);
if(checks.some(v=>v.max_absolute_error>1e-5||v.label_mismatches>0))process.exitCode=1;
