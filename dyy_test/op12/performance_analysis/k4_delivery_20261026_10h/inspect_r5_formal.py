from pathlib import Path
import argparse,json
O=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=O/'r5_graph_formal01');a=p.parse_args()
d=json.loads((a.run/'RATE_ANALYSIS.json').read_text());rows=[]
for r in d['records']:
    trials=[]
    for t in r['trials']:
        trials.append(dict(trial=t['trial'],pass_gate=t['pass_gate'],window_change=t['adjacent_window_relative_change'],slopes=[w['slope'] for w in t['windows']],R2=[w['r2'] for w in t['windows']],
          bad_points=[dict(N=p['N'],CV=p['CV'],span=p['P95_P5_over_median']) for p in t['points'] if p['CV'] is None or p['CV']>.05 or p['P95_P5_over_median']>.10]))
    rows.append(dict(configuration=r['configuration'],domain=r['domain'],corrected=r['control_corrected'],role=r['admission_role'],pass_gate=r['pass_gate'],cross_process=r['cross_process_range_over_mean'],mean_tail_slope=r['mean_tail_slope'],trials=trials))
hard=[r for r in rows if r['role']=='HARD_RAW_LOAD']
tail_only=all(not t['bad_points'] and min(t['R2'])>=.99 and min(t['slopes'])>0 for r in hard for t in r['trials']) and all(r['cross_process'] is not None and r['cross_process']<.01 for r in hard)
result=dict(status=d['status'],raw_main_tail_only_failure=tail_only and d['status']=='RETEST_REQUIRED',records=rows,model_enabled=False)
(a.run/'GATE_COMPACT.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=d['status'],raw_main_tail_only_failure=result['raw_main_tail_only_failure'],hard_records=hard),indent=2))
