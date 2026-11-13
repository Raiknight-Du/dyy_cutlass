"""Describe sampled telemetry without asserting continuous per-kernel frequency."""
from pathlib import Path
import argparse,csv,collections,json
p=argparse.ArgumentParser();p.add_argument('--telemetry',type=Path,required=True);p.add_argument('--event',type=Path,nargs='+');a=p.parse_args()
rows=[json.loads(s) for s in (a.telemetry/'SNAPSHOTS.jsonl').read_text().splitlines()]
valid=[r for r in rows if isinstance(r.get('values'),dict) and r.get('returncode')==0]
def describe(rr):
    out={'samples':len(rr),'fields':{}}
    fields=sorted({k for r in rr for k in r['values']})
    for k in fields:
        vv=[r['values'].get(k,'UNKNOWN') for r in rr]
        out['fields'][k]={'observations':dict(collections.Counter(vv))}
        if not any(k.startswith(x) for x in ['clocks_event_reasons','uuid','pstate']):
            try:out['fields'][k].update(min=min(map(float,vv)),max=max(map(float,vv)))
            except ValueError:pass
    return out
out={'all_command_snapshots':describe(valid),'model_enabled':False,
 'limitation':'Values are sampled during host query bounds. Overlap is an association, not proof of continuous frequency or reason state throughout a kernel; cumulative counters include historical activity.'}
if a.event:
    ev=[r for path in a.event for r in csv.DictReader(path.open())]
    intervals=[(int(r['host_start_time_ns']),int(r['host_end_time_ns'])) for r in ev]
    overlap=[r for r in valid if any(r['read_start_time_ns']<=end and r['read_end_time_ns']>=start for start,end in intervals)]
    out['snapshots_whose_query_bounds_overlap_sampled_Event_host_envelopes']=describe(overlap)
    out['sampled_invocations']=len(intervals)
    out['invocations_with_any_query_bounds_overlap']=sum(any(r['read_start_time_ns']<=end and r['read_end_time_ns']>=start for r in valid) for start,end in intervals)
    if not overlap:out['within_sampled_Event_frequency_reasons']='UNKNOWN'
(a.telemetry/'TELEMETRY_SUMMARY.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ['all_command_snapshots','snapshots_whose_query_bounds_overlap_sampled_Event_host_envelopes']}))
