from pathlib import Path
import argparse,csv,collections,json,statistics
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
done=json.loads((root/'completion.json').read_text());meta=json.loads((root/'ACTUAL_METADATA_GATE.json').read_text())
assert done['completed'] and done['correctness'] and done['invocations']==20 and done['full_output_elements_per_CTA']==32768
assert meta['actual_CUBIN_SHA256']=='5a3461fc33dc34bccde4b45a7b5fc55a70f68abf9cd3b453d02cef623119ba9c' and meta['CPU_text_exact_match'] and meta['max_CTA_per_SM']==1
ev=list(csv.DictReader((root/'EVENT_RAW.csv').open()));ct=list(csv.DictReader((root/'CTA_CLOCK_RAW.csv').open()));lazy=list(csv.DictReader((root/'LAZY_C1_CTA_RAW.csv').open()))
assert len(ev)==20 and len(ct)==2560 and len(lazy)==128
assert all(r['smid_begin']==r['smid_end'] and int(r['end_cycle'])-int(r['begin_cycle'])==int(r['cycles'])>0 for r in lazy)
by=collections.defaultdict(list);groups=collections.defaultdict(dict)
for r in ct:
    assert r['smid_begin']==r['smid_end'] and int(r['end_cycle'])-int(r['begin_cycle'])==int(r['cycles'])>0
    by[int(r['invocation'])].append(r)
for r in ev:
    rr=by[int(r['invocation'])];cy=[int(x['cycles']) for x in rr];sm={x['smid_begin'] for x in rr}
    assert len(rr)==128 and {int(x['cta']) for x in rr}==set(range(128)) and float(r['cta_cycle_median'])==statistics.median(cy) and float(r['cta_cycle_max'])==max(cy) and int(r['unique_SMs'])==len(sm) and r['valid']=='1'
    groups[r['config'],int(r['N'])][int(r['sample'])]=r
nodes=list((root/'GRAPH_NODES').glob('N*.json'));assert len(nodes)==4
for p in nodes:
    d=json.loads(p.read_text());n=d['native_owned_v2_edges']
    assert d['status']=='THREE_NODE_TWO_EDGE_EVENT_KERNEL_EVENT_PASS' and d['REG']==56
    assert n['count_query_returncode']==n['data_query_returncode']==n['NULL_edgeData_returncode']==0
    assert all(e['from_port']==e['to_port']==e['type']==0 and not any(e['reserved']) for e in n['edges'])
cases=[];effects=[]
for N in [128,256]:
    c1=groups['C1',N];c8=groups['C8',N];assert c1.keys()==c8.keys()==set(range(5))
    first=statistics.median(float(x['cta_cycle_median']) for x in c1.values());second=statistics.median(float(x['cta_cycle_median']) for x in c8.values())
    paired=[100*(float(c8[s]['cta_cycle_median'])/float(c1[s]['cta_cycle_median'])-1) for s in sorted(c1)]
    effects.append(dict(N=N,C1_native_cycles=first,C8_native_cycles=second,ratio_of_medians_change_pct=100*(second/first-1),paired_change_pct_median=statistics.median(paired),paired_change_pct_samples=paired))
    for config,rows in [('C1',c1),('C8',c8)]:cases.append(dict(configuration=config,N=N,Event_us_median=statistics.median(float(x['event_us']) for x in rows.values()),native_cycles_median=statistics.median(float(x['cta_cycle_median']) for x in rows.values()),unique_SM_min=min(int(x['unique_SMs']) for x in rows.values()),unique_SM_max=max(int(x['unique_SMs']) for x in rows.values())))
negative=all(abs(e['ratio_of_medians_change_pct'])<3 and abs(e['paired_change_pct_median'])<3 for e in effects)
result=dict(status='NEGATIVE_FINITE_CONTROL_STOP_FURTHER_GPU' if negative else 'SIGNIFICANT_OR_AMBIGUOUS_RETURN_TO_LEAD',effects=effects,cases=cases,
  raw_gate='20calls/2560CTA/128lazy/4graphs and32768gold_elements_per_CTA passed',stop_further_GPU=negative,
  limits='Native internal cycles are primary. Whole Event includes common16MiB output with different values; no independent MMA benefit inferred. Screen is not a formal stable rate.',model_enabled=False)
(root/'SCREEN_SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
