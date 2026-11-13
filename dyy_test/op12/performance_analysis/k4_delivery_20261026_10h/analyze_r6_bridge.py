from pathlib import Path
import csv,collections,json,re
O=Path(__file__).resolve().parent;root=O/'r6_ncu01';cases={}
def number(v):return float(v.replace(',','')) if v and v not in ['n/a','N/A','-'] else 0
for role in ['B0','V0']:
    run=root/role;raw=list(csv.reader((run/(role+'_raw.csv')).open()));assert len(raw)==3
    metrics=dict(zip(raw[0],raw[2]));src=list(csv.reader((run/(role+'_source.csv')).open()));h=src[1]
    rows=[dict(zip(h,r)) for r in src[2:] if len(r)==len(h) and r[0].startswith('0x')];base=int(rows[0]['Address'],16)
    state=collections.defaultdict(list);inst=collections.Counter();pred=collections.Counter();sync=[]
    for r in rows:
        pc=int(r['Address'],16)-base;text=r['Source'].strip();op=re.sub(r'^@!?\w+\s+','',text).split()[0].split('.')[0]
        inst[op]+=number(r.get('Instructions Executed'));pred[op]+=number(r.get('Predicated-On Thread Instructions Executed'))
        if (0x5dd0<=pc<=0x65c0 if role=='B0' else 0x5dd0<=pc<=0x5fc0):state['fixed'].append(r)
        if (0x6a70<=pc<=0x6c60 if role=='B0' else 0x6470<=pc<=0x6660):state['initial'].append(r)
        if any(x in text for x in ['SYNCS','TRYWAIT']):sync.append(dict(PC=hex(pc),source=text,warp_instructions=number(r.get('Instructions Executed'))))
    sums={}
    for typ,rr in state.items():
        assert len(rr)==(128 if role=='B0' and typ=='fixed' else 32) and all('LDG.E.NA' in r['Source'] for r in rr)
        sums[typ]={k:sum(number(r.get(k)) for r in rr) for k in ['Instructions Executed','Predicated-On Thread Instructions Executed','L2 Theoretical Sectors Global','L2 Theoretical Sectors Global Ideal']}
        sums[typ]['PCs']=len(rr);sums[typ]['request_B']=int(sums[typ]['L2 Theoretical Sectors Global']*32)
    assert sums['initial']['request_B']==2097152 and sums['fixed']['request_B']==(58720256 if role=='B0' else 14680064)
    cases[role]=dict(metrics=metrics,state_PC=sums,issued_opcode_counts=dict(inst),pred_on_opcode_counts=dict(pred),sync_inventory=sync)
invariants={op:cases['B0']['issued_opcode_counts'].get(op)==cases['V0']['issued_opcode_counts'].get(op) for op in ['UTCHMMA','UTMALDG','UTMASTG','LDTM','STTM','LDL','STL','STG']}
assert all(invariants.values()),invariants
metric_names=sorted(set(cases['B0']['metrics'])|set(cases['V0']['metrics']))
deltas={k:dict(B0=number(cases['B0']['metrics'].get(k)),V0=number(cases['V0']['metrics'].get(k)),B0_minus_V0=number(cases['B0']['metrics'].get(k))-number(cases['V0']['metrics'].get(k))) for k in metric_names if k.endswith('.sum') and any(x in k for x in ['l1tex__data','l1tex__t_','lts__','dram__','l1tex__m_'])}
assert deltas['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum']['B0_minus_V0']==1376256
observed=deltas['l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_read.sum']['B0_minus_V0']
result=dict(status='ORIGINAL_NATIVE_SOURCE_WORKLOAD_GATE_PASS',cases=cases,native_workload_invariants=invariants,metric_deltas=deltas,
 prediction=dict(expected_Data_read_difference=1376256,observed_Data_read_difference=observed,residual=observed-1376256,exact_match=observed==1376256,model_input=False),
 limits=['Full K4 action counters across application replay passes; no Data/conflict-to-cycle conversion or direct INIT elapsed attribution.','Source address work, hardware bank arbitration and physical cache requests remain separate scopes.','No new performance samples, kernel variant, mapping adjustment or model parameter.'],model_enabled=False)
(root/'BRIDGE_COUNTER_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],prediction=result['prediction'],invariants=invariants)))
