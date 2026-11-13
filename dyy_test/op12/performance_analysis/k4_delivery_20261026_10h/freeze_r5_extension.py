from pathlib import Path
import csv,collections,json,statistics,subprocess,sys
O=Path(__file__).resolve().parent;root=O/'r5_graph_formal_ext16384_01';records=[]
for trial in range(3):
    run=root/f'trial{trial}';done=json.loads((run/'completion.json').read_text());assert done['completed'] and done['correctness'] and done['invocations']==960 and done['samples']==30
    for source,sha in [('SS','60b132f31202059cf0d7a29a49b2097f0c4109909de97a4645cf73706cd08038'),('TS','34e6c4bdffb18e076cf1d0809a6645277a85c1e7e1a60ec47b92e1a1fb0b5ebf')]:
        m=json.loads((run/('ACTUAL_METADATA_GATE_'+source+'.json')).read_text());assert m['actual_CUBIN_SHA256']==sha and m['CPU_text_exact_match'] and m['max_CTA_per_SM']==1
    ev=list(csv.DictReader((run/'EVENT_RAW.csv').open()));ct=list(csv.DictReader((run/'CTA_CLOCK_RAW.csv').open()));lazy=list(csv.DictReader((run/'LAZY_CONTROL_CTA_RAW.csv').open()))
    assert len(ev)==960 and len(ct)==122880 and len(lazy)==256
    for r in lazy:assert r['smid_begin']==r['smid_end'] and int(r['end_cycle'])-int(r['begin_cycle'])==int(r['cycles'])>0
    by=collections.defaultdict(list)
    for r in ct:
        assert r['smid_begin']==r['smid_end'] and int(r['end_cycle'])-int(r['begin_cycle'])==int(r['cycles'])>0
        by[int(r['invocation'])].append(r)
    sms=collections.Counter()
    for r in ev:
        rr=by[int(r['invocation'])];cy=[int(x['cycles']) for x in rr];sm={x['smid_begin'] for x in rr}
        assert len(rr)==128 and {int(x['cta']) for x in rr}==set(range(128)) and float(r['cta_cycle_median'])==statistics.median(cy) and float(r['cta_cycle_max'])==max(cy) and int(r['unique_SMs'])==len(sm) and r['valid']=='1'
        sms[len(sm)]+=1
    nodes=list((run/'GRAPH_NODES').glob('N*.json'));assert len(nodes)==32
    for p in nodes:
        d=json.loads(p.read_text());assert d['status']=='THREE_NODE_TWO_EDGE_EVENT_KERNEL_EVENT_PASS'
        n=d['native_owned_v2_edges'];assert n['count_query_returncode']==n['data_query_returncode']==n['NULL_edgeData_returncode']==0
        assert all(e['from_port']==e['to_port']==e['type']==0 and not any(e['reserved']) for e in n['edges'])
    records.append(dict(trial=trial,launches=960,CTA_records=len(ct),lazy_CONTROL_CTA_records=len(lazy),graph_certificates=len(nodes),unique_SM_hist=dict(sms)))
(root/'RAW_TOPOLOGY_AUDIT.json').write_text(json.dumps(dict(status='ALL2880_LAUNCH_368640CTA_768LAZY_96GRAPH_VALID',records=records,model_enabled=False),indent=2))
subprocess.run([sys.executable,str(O/'analyze_reason_telemetry.py'),'--telemetry',str(O/'r5_graph_formal_ext16384_01_telemetry')],check=True)
subprocess.run([sys.executable,str(O/'pack_evidence.py'),'--name','R5_GRAPH_FORMAL_EXT16384_01_EVIDENCE','--items',root.name,'r5_graph_formal_ext16384_01_telemetry','guard_r5_graph_formal_ext16384_01','R5_GRAPH_FORMAL_EXT_ADMISSION.json','R5_GRAPH_FORMAL_EXT_COMMAND.json','R5_GRAPH_FORMAL_EXT_BATCH_COMMAND.json','R5_GRAPH_FORMAL_EXT16384_01_AUTO_RESTORE_VERIFICATION.json'],check=True)
