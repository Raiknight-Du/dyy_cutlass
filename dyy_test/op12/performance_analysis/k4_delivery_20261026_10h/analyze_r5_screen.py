from pathlib import Path
import argparse,csv,collections,json,statistics
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
done=json.loads((root/'completion.json').read_text());assert done['completed'] and done['correctness'] and done['invocations']==40 and done['full_output_elements_per_CTA']==4096
expected={'SS':('60b132f31202059cf0d7a29a49b2097f0c4109909de97a4645cf73706cd08038',52),'TS':('34e6c4bdffb18e076cf1d0809a6645277a85c1e7e1a60ec47b92e1a1fb0b5ebf',158)}
meta={}
for source,(sha,reg) in expected.items():
    m=json.loads((root/('ACTUAL_METADATA_GATE_'+source+'.json')).read_text());assert m['actual_CUBIN_SHA256']==sha and m['CPU_text_exact_match'] and m['max_CTA_per_SM']==1 and m['helper_invoked']
    meta[source]=m
ev=list(csv.DictReader((root/'EVENT_RAW.csv').open()));ct=list(csv.DictReader((root/'CTA_CLOCK_RAW.csv').open()));lazy=list(csv.DictReader((root/'LAZY_CONTROL_CTA_RAW.csv').open()))
assert len(ev)==40 and len(ct)==5120 and len(lazy)==256
for source in expected:
    rr=[r for r in lazy if r['source']==source];assert len(rr)==128 and {int(r['cta']) for r in rr}==set(range(128))
    assert all(r['mode']=='CONTROL' and int(r['N'])==128 and int(r['end_cycle'])-int(r['begin_cycle'])==int(r['cycles'])>0 and r['smid_begin']==r['smid_end'] for r in rr)
by=collections.defaultdict(list)
for r in ct:
    assert r['smid_begin']==r['smid_end'] and int(r['end_cycle'])-int(r['begin_cycle'])==int(r['cycles'])>0
    by[int(r['invocation'])].append(r)
groups=collections.defaultdict(list)
for r in ev:
    rr=by[int(r['invocation'])];cy=[int(x['cycles']) for x in rr];sm={x['smid_begin'] for x in rr}
    assert len(rr)==128 and {int(x['cta']) for x in rr}==set(range(128)) and float(r['cta_cycle_median'])==statistics.median(cy) and float(r['cta_cycle_max'])==max(cy) and int(r['unique_SMs'])==len(sm) and r['valid']=='1'
    groups[r['config'],int(r['N'])].append(r)
nodes=list((root/'GRAPH_NODES').glob('*.json'));assert len(nodes)==8
for p in nodes:
    d=json.loads(p.read_text());assert d['status']=='THREE_NODE_TWO_EDGE_EVENT_KERNEL_EVENT_PASS' and d['REG'] in [52,158]
    n=d['native_owned_v2_edges'];assert n['count_query_returncode']==n['data_query_returncode']==n['NULL_edgeData_returncode']==0
    assert all(e['from_port']==e['to_port']==e['type']==0 and not any(e['reserved']) for e in n['edges'])
cases=[]
for (config,N),rr in sorted(groups.items()):
    assert len(rr)==5
    cases.append(dict(configuration=config,N=N,Event_us_median=statistics.median(float(x['event_us']) for x in rr),CTA_cycle_median=statistics.median(float(x['cta_cycle_median']) for x in rr),unique_SM_min=min(int(x['unique_SMs']) for x in rr),unique_SM_max=max(int(x['unique_SMs']) for x in rr)))
result=dict(status='SCREEN40CALL_5120CTA_256LAZY_8GRAPH_ALL_PASS_PENDING_LEAD',cases=cases,metadata=meta,completion=done,
 main_rate_gate='Only raw SS_LOAD/TS_LOAD Event/native; CONTROL has no N-dependent MMA loop; sensitivity subtraction diagnostic only',model_enabled=False)
(root/'SCREEN_SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],cases=cases),indent=2))
