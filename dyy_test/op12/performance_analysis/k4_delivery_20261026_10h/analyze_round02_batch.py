"""Summarize per-process return envelopes without fitting a theoretical intercept."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
rows=[];records=[]
for S in [8192,16384]:
    for trial in range(3):
        r=a.run/f'S{S}'/f'trial{trial}'
        e=json.loads((r/'EVENT_QC.json').read_text());d=json.loads((r/'ENDPOINT_SUMMARY.json').read_text())
        v=d['summary']['K4_EVENT'];q=e['K4_EVENT_REF__CLOCK_RETURNS'];cp=e['CP_EVENT_REF__CLOCK_RETURNS']
        row=dict(S=S,C=S//8,L=S//512,trial=trial,valid_paired_calls=d['valid_calls'],events_per_call=d['expected_events'],
            CP_Event_probe_change_pct=cp['paired_change_pct'],CP_Event_CI_low=cp['CI95_pct'][0],CP_Event_CI_high=cp['CI95_pct'][1],
            K4_Event_probe_change_pct=q['paired_change_pct'],K4_Event_CI_low=q['CI95_pct'][0],K4_Event_CI_high=q['CI95_pct'][1],
            **{k+'_median':x['median'] for k,x in v.items()})
        rows.append(row);records.append(dict(S=S,trial=trial,event=e,endpoint=d))
        assert cp['equivalent'] and q['equivalent']
with (a.run/'FORMAL_ENDPOINT_TABLE.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(status='FORMAL_GATES_PASSED',shapes=[8192,16384],independent_processes_per_shape=3,
    total_paired_calls=sum(x['valid_paired_calls'] for x in rows),records=records,
    local_absolute_bias_bound_cycles=None,model_enabled=False,
    interpretation='Per-process distributions from same-call endpoints; no mean/median stitching into one trace. Return intervals include waits and handoff. Mid indices are predeclared, not a changed model boundary. No fitted rate/intercept is adopted.')
(a.run/'FORMAL_SUMMARY.json').write_text(json.dumps(report,indent=2))
print(json.dumps(rows,indent=2))
