import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const root=path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
const folder=path.join(root,'static_artifacts','binary');
const records=[];
for(const f of fs.readdirSync(folder).filter(x=>x.endsWith('.sass'))){
  const text=fs.readFileSync(path.join(folder,f),'utf8');
  const k=f.includes('CPDeltaRuleTPrecompute')?'K1':f.includes('CPDeltaRuleMNPrecompute')?'K2':f.includes('CPDeltaRuleFixup')?'K3':'K4';
  const sites={};
  for(const m of text.matchAll(/\/\*\s*[0-9a-fA-F]+\s*\*\/\s*(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)/g)){
    const op=m[1];if(/HMMA|UTCHMMA|UTCQMMA|UTC|UTMA|LDSM|CPASYNC|DEPBAR|MEMBAR|BAR|F2F|FFMA|SYNCS/.test(op))sites[op]=(sites[op]??0)+1;
  }
  records.push({kernel:k,file:f,sha256:crypto.createHash('sha256').update(text).digest('hex'),static_instruction_sites:sites,warning:'Static SASS sites are not dynamic instruction counts. Loops, predicates and atom grouping must be analyzed separately. No timing was collected.'});
}
records.sort((a,b)=>a.kernel.localeCompare(b.kernel));
fs.writeFileSync(path.join(root,'static_artifacts','sass_audit.json'),JSON.stringify(records,null,2));
console.log(JSON.stringify(records,null,2));
