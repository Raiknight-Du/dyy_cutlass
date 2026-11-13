from pathlib import Path
import argparse,csv,json,re
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--engineering',type=Path,required=True);a=p.parse_args()
s=list(csv.reader((a.run/(a.config+'_source.csv')).open()));h=s[1]
rows=[dict(zip(h,r)) for r in s[2:] if len(r)==len(h) and r[0].startswith('0x')];base=int(rows[0]['Address'],16)
expected={int(r['pc'],16):r for r in csv.DictReader((a.engineering/'ACTUAL_SASS_SCOPE.csv').open(encoding='utf-8-sig'))}
labels={};pending=[]
for line in next((a.engineering/'compile01_venv').glob('*.sass')).read_text().splitlines():
    m=re.match(r'^([\w.$]+):',line.strip())
    if m:pending.append(m[1])
    pc=re.search(r'/\*([0-9a-f]+)\*/',line)
    if pc:
        for name in pending:labels[name]=int(pc[1],16)
        pending=[]
def norm(x,static=False):
    x=x.replace('.reuse','')
    if static:
        x=re.sub(r'(\bR\d+)\s+`',r'\1, `',x)
        x=re.sub(r'`\(([^)]+)\)',lambda m:hex(labels[m[1]]),x)
    else:x=re.sub(r'`\([^)]*\)','',x)
    if re.search(r'\b(?:BRA|BSSY|CALL|RET)',x):x=re.sub(r'0x([0-9a-f]+)',lambda m:hex(int(m[1],16)-base) if int(m[1],16)>=base else m[0],x)
    return re.sub(r'\s+','',x.strip().rstrip(';'))
diff=[]
for r in rows:
    pc=int(r['Address'],16)-base;e=expected.get(pc)
    text=((e['predicate']+' ') if e and e['predicate'] else '')+(e['opcode']+' '+e['operands'] if e else '')
    if not e or norm(text,True)!=norm(r['Source']):diff.append(dict(pc=hex(pc),expected=text,exported=r['Source']))
raw=list(csv.reader((a.run/(a.config+'_raw.csv')).open()));assert len(raw)==3
metrics=dict(zip(raw[0],raw[2]));assert metrics['launch__registers_per_thread']=='79'
assert metrics['launch__shared_mem_per_block_dynamic']=='225280'
metadata=[]
for f in (a.run/'replays').glob('*/ACTUAL_METADATA_GATE.json'):
    m=json.loads(f.read_text());assert m['actual_CUBIN_SHA256']=='5ebdc94be5cda307ee64cb975209e11ed68ef25a420cfc3598b00c15368cc066'
    assert m['CPU_text_exact_match'] and m['max_CTA_per_SM']==1 and m['helper_invoked']
    metadata.append(str(f.relative_to(a.run)))
assert metadata and len(rows)==len(expected) and not diff,diff
assert not any(re.search(r'\b(?:LDL|STL)\b',r['Source']) for r in rows)
result=dict(status='ACTUAL_NATIVE_IDENTITY_RESOURCE_PASS',exported_PCs=len(rows),native_PCs=len(expected),metadata_replays_checked=len(metadata),
 SASS_mismatches=diff,normalization='whitespace, displayed .reuse hints, optional branch labels and absolute branch relocation only',REG=79,LDL=0,STL=0,dynamic_SMEM_B=225280,model_enabled=False)
(a.run/'IDENTITY_RESOURCE_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
