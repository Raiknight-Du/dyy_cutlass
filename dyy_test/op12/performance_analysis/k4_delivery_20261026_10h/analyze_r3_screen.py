"""Screen identity/correctness/SM mapping audit, no throughput admission."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
root=a.run/'trial0'
done=json.loads((root/'completion.json').read_text());meta=json.loads((root/'runtime_metadata.json').read_text())
assert done['completed'] and done['correctness'] and done['mode']=='screen'
assert done['invocations']==80 and done['warmups_per_config_per_point']==20 and done['samples_per_point']==5
ev=list(csv.DictReader((root/'EVENT_RAW.csv').open()));ct=list(csv.DictReader((root/'CTA_CLOCK_RAW.csv').open()))
assert len(ev)==80 and len(ct)==5160
by=collections.defaultdict(list)
for r in ct:
    assert int(r['smid_begin'])==int(r['smid_end'])
    assert int(r['end_cycle'])-int(r['begin_cycle'])==int(r['cycles'])>0
    by[int(r['invocation'])].append(r)
assert len(by)==80
groups=collections.defaultdict(list);maps=[]
for r in ev:
    assert r['valid']=='1'
    grid=int(r['config'].split('_')[1][1:]);load=r['config'].endswith('_LOAD');vector=r['config'].startswith('V')
    inv=int(r['invocation']);N=int(r['N']);assert N in [128,256]
    assert int(r['logical_B'])==(N*grid*65536 if load else 0)
    assert int(r['theoretical_request_B'])==(N*grid*(131072 if vector else 524288) if load else 0)
    rows=by[inv];assert len(rows)==grid and {int(x['cta']) for x in rows}==set(range(grid))
    cc=[int(x['cycles']) for x in rows];sm=collections.Counter(int(x['smid_begin']) for x in rows)
    assert int(r['unique_SMs'])==len(sm)
    assert float(r['cta_cycle_median'])==statistics.median(cc)
    assert float(r['cta_cycle_max'])==max(cc)
    groups[r['config'],N].append(r)
    maps.append(dict(invocation=inv,config=r['config'],N=N,SMID_CTA_counts=dict(sm),max_CTAs_on_one_SM=max(sm.values()),unique_SMs=len(sm)))
def quant(xs,q):
    x=sorted(xs);i=q*(len(x)-1);lo=int(i);hi=min(lo+1,len(x)-1);return x[lo]+(x[hi]-x[lo])*(i-lo)
def stats(xs):
    return dict(n=len(xs),median=statistics.median(xs),min=min(xs),max=max(xs),CV_pct=statistics.pstdev(xs)/statistics.mean(xs)*100,
       P95_P5_over_median_pct=(quant(xs,.95)-quant(xs,.05))/statistics.median(xs)*100)
summary=[]
for (config,N),rows in sorted(groups.items()):
    assert len(rows)==5
    summary.append(dict(config=config,N=N,Event_us=stats([float(x['event_us']) for x in rows]),
       per_launch_CTA_cycle_median=stats([float(x['cta_cycle_median']) for x in rows]),
       unique_SM_min=min(int(x['unique_SMs']) for x in rows),unique_SM_max=max(int(x['unique_SMs']) for x in rows)))
report=dict(status='SCREEN_RAW_VALIDATED_PENDING_LEAD_REVIEW',metadata=meta,completion=done,invocations=len(ev),CTA_records=len(ct),
     cases=summary,frequency_policy='Automatic; R3 did not lock GPU clocks',model_enabled=False,
     interpretation='Two work points and one process are diagnostic only; no fitted conditional service or hardware capacity admitted. Per-CTA durations never subtract different-SM clocks. Shared residency cap does not guarantee simultaneous execution.')
(a.run/'SCREEN_SUMMARY.json').write_text(json.dumps(report,indent=2))
(a.run/'SMID_MAPPING_AUDIT.json').write_text(json.dumps(maps,indent=2))
print(json.dumps(report,indent=2))
