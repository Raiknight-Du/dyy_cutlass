"""Single scalar N512 action audit; no replay timing used as service timing."""
from pathlib import Path
import argparse,csv,collections,json,re
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--engineering',type=Path,required=True);p.add_argument('--config',default='S_G128_LOAD');a=p.parse_args()
vector=a.config.startswith('V');grid=int(a.config.split('_')[1][1:]);active=a.config.endswith('_LOAD');N=512
expected_warp=4*N*grid if active else 0;expected_lanes=128*N*grid if active else 0
raw=list(csv.reader((a.run/(a.config+'_raw.csv')).open()));assert len(raw)==3
metrics=dict(zip(raw[0],raw[2]));units=dict(zip(raw[0],raw[1]))
s=list(csv.reader((a.run/(a.config+'_source.csv')).open()));h=s[1]
data=[dict(zip(h,r)) for r in s[2:] if len(r)==len(h) and r[0].startswith('0x')]
base=int(data[0]['Address'],16)
def num(x):return int(float(x.replace(',',''))) if x not in ['','n/a','N/A','nan'] else None
def norm(x):
    x=x.replace('.reuse','')
    x=re.sub(r'`\([^)]*\)','',x)
    if re.search(r'\bBRA',x):
        x=re.sub(r'0x([0-9a-f]+)',lambda m:hex(int(m[1],16)-base) if int(m[1],16)>=base else m[0],x)
    return re.sub(r'\s+','',x.rstrip(';').strip())
expected=list(csv.DictReader((a.engineering/'ACTUAL_SASS_SCOPE.csv').open(encoding='utf-8-sig')))
expected={int(r['pc'],16):r for r in expected if r['variant']==('vector' if vector else 'scalar')}
observed={int(r['Address'],16)-base:r for r in data}
mismatches=[]
for pc,r in observed.items():
    if not r.get('Source','').strip():continue
    e=expected.get(pc)
    text=((e['predicate']+' ') if e and e['predicate'] else '')+(e['opcode']+' '+e['operands'] if e else '')
    if e is None or norm(text)!=norm(r['Source']):mismatches.append({'PC':hex(pc),'expected':text,'exported':r['Source']})
loads=[(pc,r) for pc,r in observed.items() if re.search(r'\bLDG',r['Source'])]
assert len(loads)==(32 if vector else 128)
details=[];totals=collections.Counter();opcounts=collections.Counter()
for pc,r in observed.items():
    op=re.sub(r'^@!?\w+\s+','',r['Source'].strip()).split()[0] if r['Source'].strip() else 'EMPTY'
    opcounts[op]+=num(r['Instructions Executed']) or 0
for pc,r in loads:
    values={k:num(r.get(k,'')) for k in ['Instructions Executed','Predicated-On Thread Instructions Executed','L2 Theoretical Sectors Global','L2 Theoretical Sectors Local']}
    assert (values['Instructions Executed'] or 0)==expected_warp,(hex(pc),values)
    assert (values['Predicated-On Thread Instructions Executed'] or 0)==expected_lanes,(hex(pc),values)
    assert (values['L2 Theoretical Sectors Global'] or 0)==expected_warp*32,(hex(pc),values)
    assert not values['L2 Theoretical Sectors Local']
    for k,v in values.items():totals[k]+=v or 0
    details.append({'PC':hex(pc),'source':r['Source'],**values})
assert totals['Instructions Executed']==expected_warp*len(loads) and totals['L2 Theoretical Sectors Global']==expected_warp*len(loads)*32
assert not any(k.startswith(('LDL','STL')) and v for k,v in opcounts.items())
if 'L2 Theoretical Sectors Local' not in h:totals['L2 Theoretical Sectors Local']=None
if 'L2 Theoretical Sectors Global' not in h:totals['L2 Theoretical Sectors Global']=None
report={'status':'DYNAMIC_COUNTS_PASS_SASS_MATCH' if not mismatches else 'DYNAMIC_COUNTS_PASS_SASS_REVIEW_REQUIRED',
 'configuration':a.config,'load_PC_count':len(loads),'load_totals':dict(totals),'logical_B':N*grid*65536 if active else 0,'expected_Source_request_B':N*grid*65536*(2 if vector else 8) if active else 0,
 'native_SASS_comparison_mismatches':mismatches,'observed_SASS_PC_count':len(observed),'static_SASS_PC_count':len(expected),
 'SASS_normalization':'Whitespace, displayed .reuse hint, optional branch label and absolute-to-relative branch targets only; opcodes/registers/predicates/immediates otherwise compared.',
 'Source_local_sector_column_present':'L2 Theoretical Sectors Local' in h,
 'full_action_issued_opcode_counts':dict(opcounts),'metrics':metrics,'units':units,'load_PC_details':details,
 'interpretation':'Counters belong to one selected action across application replay passes. Full-action fixed STG/const items are separate from loop LDG. Replay durations are not timing evidence. Native-only app and exported SASS comparison constrain app identity; no rate/model adoption.',
 'model_enabled':False}
(a.run/'COUNTER_AUDIT.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['metrics','units','load_PC_details','full_action_issued_opcode_counts']}))
