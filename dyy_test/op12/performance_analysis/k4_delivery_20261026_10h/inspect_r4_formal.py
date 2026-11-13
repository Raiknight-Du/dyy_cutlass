from pathlib import Path
import json,argparse
O=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=O/'r4_formal01');a=p.parse_args();run=a.run;d=json.loads((run/'RATE_ANALYSIS.json').read_text())
rows=[]
for r in d['records']:
    trials=[]
    for t in r['trials']:
        trials.append(dict(trial=t['trial'],pass_gate=t['pass_gate'],window_change=t['adjacent_window_relative_change'],
            bad_points=[dict(N=p['N'],CV=p['CV'],spread=p['P95_P5_over_median']) for p in t['points'] if p['CV'] is None or p['CV']>.05 or p['P95_P5_over_median']>.10],
            slopes=[w['slope'] for w in t['windows']],R2=[w['r2'] for w in t['windows']]))
    rows.append(dict(configuration=r['configuration'],domain=r['domain'],corrected=r['control_corrected'],role=r['admission_role'],pass_gate=r['pass_gate'],cross_process=r['cross_process_range_over_mean'],trials=trials))
(run/'GATE_COMPACT.json').write_text(json.dumps(dict(status=d['status'],records=rows),indent=2));print(json.dumps(dict(status=d['status'],records=[r for r in rows if not r['pass_gate']]),indent=2))
