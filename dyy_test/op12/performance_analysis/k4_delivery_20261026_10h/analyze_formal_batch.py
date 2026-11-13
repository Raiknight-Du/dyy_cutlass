"""Keep processes independent when applying the frozen effect threshold."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
records=[]
for trial in range(3):
    r=a.run/f'trial{trial}'
    d=json.loads((r/'OFFLINE_ANALYSIS.json').read_text())
    records.append(dict(trial=trial,raw_calls_checked=d['raw_calls_checked'],event=d['event'],paired_INIT=d['paired_INIT'],cycles=d['cycles'],correct=d['correct']))
effect={}
for scope in ['CP_EVENT','K4_EVENT']:
    row=[x['paired_INIT'][scope+'_FRAME0__FRAME7'] for x in records]
    values=[x['paired_change_pct'] for x in row]
    sd=statistics.stdev(values);threshold=max(3,2*sd);m=statistics.median(values)
    same_sign=all(v<0 for v in values) or all(v>0 for v in values)
    exclude_zero=all(ci['CI95_pct'][0]>0 or ci['CI95_pct'][1]<0 for ci in row)
    effect[scope]=dict(process_paired_change_pct=values,process_SD_pct=sd,predeclared_threshold_pct=threshold,
         median_process_effect_pct=m,all_processes_same_sign=same_sign,all_process_CIs_exclude_zero=exclude_zero,
         stable_main_INIT_effect=abs(m)>threshold and same_sign and exclude_zero)
report=dict(status='THREE_PROCESSES_OFFLINE_AUDITED',independent_processes=3,records=records,effect=effect,
    total_paired_raw_calls=sum(x['raw_calls_checked'] for x in records),
    local_absolute_bias_bound_cycles=None,model_enabled=False,
    interpretation='Net runtime gate effect includes added waits/polling. It is not an absolute upper bound on competition. INIT shortening can move waits into startup; Event and first-return bundle must be examined independently. No frequency-based conversion.')
(a.run/'FORMAL_SUMMARY.json').write_text(json.dumps(report,indent=2))
compact=dict(effect=effect,controls=[dict(trial=r['trial'],event={s:r['event'][s+'_REF__FRAME0'] for s in ['CP_EVENT','K4_EVENT']},
           local={s:r['paired_INIT'][s+'_PBASE__FRAME0'] for s in ['CP_EVENT','K4_EVENT']}) for r in records])
print(json.dumps(compact,indent=2))
