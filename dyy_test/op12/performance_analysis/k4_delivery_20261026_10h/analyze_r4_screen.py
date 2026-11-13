"""Rebuild R4 same-CTA screen statistics from all raw endpoint rows."""
from pathlib import Path
import argparse,csv,collections,json,statistics
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
o=a.run;done=json.loads((o/'completion.json').read_text());meta=json.loads((o/'ACTUAL_METADATA_GATE.json').read_text())
assert done['completed'] and done['correctness'] and done['empty_commit_modes_returned'] and done['invocations']==40
assert meta['CPU_text_exact_match'] and meta['max_CTA_per_SM']==1 and meta['helper_invoked']
ev=list(csv.DictReader((o/'EVENT_RAW.csv').open()));ct=list(csv.DictReader((o/'CTA_CLOCK_RAW.csv').open()))
assert len(ev)==40 and len(ct)==5120
by=collections.defaultdict(list)
for r in ct:
    assert int(r['smid_begin'])==int(r['smid_end']) and int(r['end_cycle'])-int(r['begin_cycle'])==int(r['cycles'])>0
    by[int(r['invocation'])].append(r)
groups=collections.defaultdict(list);maps=[]
for r in ev:
    rows=by[int(r['invocation'])];assert len(rows)==128 and {int(x['cta']) for x in rows}==set(range(128))
    cycles=[int(x['cycles']) for x in rows];sms=collections.Counter(int(x['smid_begin']) for x in rows)
    assert r['valid']=='1' and float(r['cta_cycle_median'])==statistics.median(cycles) and float(r['cta_cycle_max'])==max(cycles)
    assert int(r['unique_SMs'])==len(sms);groups[r['config'],int(r['N'])].append(r)
    maps.append(dict(invocation=int(r['invocation']),config=r['config'],N=int(r['N']),SM_CTA_counts=dict(sms)))
cases=[]
for (config,n),rows in sorted(groups.items()):
    assert len(rows)==5
    cases.append(dict(config=config,N=n,event_us_median=statistics.median(float(r['event_us']) for r in rows),
     CTA_cycle_median=statistics.median(float(r['cta_cycle_median']) for r in rows),
     unique_SM_min=min(int(r['unique_SMs']) for r in rows),unique_SM_max=max(int(r['unique_SMs']) for r in rows)))
summary=dict(status='SCREEN_RAW_VALIDATED_PENDING_LEAD_REVIEW',invocations=40,CTA_records=5120,
  empty_commit_modes_returned=True,cases=cases,actual_metadata=meta,clock_policy='automatic; screen separate from R3 formal 1845 request',
  interpretation='Per-launch median of same-CTA durations. Short CONTROL may use fewer SMs; occupancy cap is not a concurrency guarantee. Fixed prefix/notification/poll remain in common envelope. No stable rate or model admission.',model_enabled=False)
(o/'SCREEN_SUMMARY.json').write_text(json.dumps(summary,indent=2));(o/'SMID_MAPPING_AUDIT.json').write_text(json.dumps(maps,indent=2))
print(json.dumps(summary,indent=2))
