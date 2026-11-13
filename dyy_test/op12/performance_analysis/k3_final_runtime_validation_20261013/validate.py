"""Independent raw Event/slopes and artifact checks, not model calibration."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

root=Path(__file__).resolve().parent
analysis=root/sys.argv[1]/'RESULTS.json'
data=json.loads(analysis.read_text())
ncu_expected=int(sys.argv[2]) if len(sys.argv)>2 else 6
checks=[]
def add(name,ok,details=None):
    checks.append({'name':name,'passed':bool(ok),'details':details})
for phase,tag in data['formal_tags'].items():
    ledger=json.loads((root/'runs'/tag/'RUN_LEDGER.json').read_text())
    add(phase+'_completed',ledger['status']=='COMPLETED')
    add(phase+'_clock_valid',ledger['frequency_valid'] and not ledger['competition'])
    add(phase+'_clock_reset',ledger['reset']['returncode']==0)
    hist={}
    for line in (root/'runs'/tag/'frequency.jsonl').open():
        r=json.loads(line); hist[r['sm_mhz']]=hist.get(r['sm_mhz'],0)+1
    add(phase+'_raw_frequency_1845',set(hist)=={1845},hist)

for trial in (1,2,3):
    folder=root/'runs'/data['formal_tags']['event']/('event_t'+str(trial))
    correct=json.loads((folder/'correctness.json').read_text())
    add('event'+str(trial)+'_all_modes_correct',len(correct)==6 and all(c['passed'] for c in correct),correct)
    rows=list(csv.DictReader((folder/'event_samples.csv').open()))
    add('event'+str(trial)+'_raw_count',len(rows)==4800,len(rows))
    for summary in json.loads((folder/'event_summary.json').read_text()):
        style,mode=summary['target']; label=style+'_M'+str(mode)
        rs=[r for r in rows if r['method']==summary['method'] and r['comparison']==label and r['reference']==summary['reference']]
        byarm={a:{int(r['pair']):float(r['us']) for r in rs if r['arm']==a} for a in ('BASE','VAR')}
        delta=statistics.median(byarm['BASE'][i]-byarm['VAR'][i] for i in byarm['BASE'])
        add('event_delta_recalc_'+str(trial)+'_'+summary['method']+'_'+label+'_'+summary['reference'],
            abs(delta-summary['paired_median_gain_us'])<1e-12)
    # These hashes are one artifact per style, not one artifact per mode.
    add('one_binary_per_style_'+str(trial),all(len(list((folder/'binary'/s).glob('*.cubin')))==1 for s in {c['style'] for c in correct}))

for gate in data['slopes']['strict_convergence_gate']:
    add('strict_gate_'+gate['style']+'_M'+str(gate['mode']),gate['status']=='PASS',gate)
for trial in (1,2,3):
    folder=root/'runs'/data['formal_tags']['slope']/('slope_t'+str(trial))
    correct=json.loads((folder/'micro_correctness.json').read_text())
    add('micro_correct_'+str(trial),len(correct)==6 and all(c['omission_correct'] and c['fixed_max_abs'] in (0,None) and c['final_max_abs'] in (0,None) for c in correct))
    raw=list(csv.DictReader((folder/'slope_samples.csv').open()))
    add('micro_raw_count_'+str(trial),len(raw)==3200,len(raw))
    for curve in [c for c in data['slopes']['curves'] if c['trial']==trial]:
        for point in curve['points']:
            xs=[float(r['us']) for r in raw if r['style']==curve['style'] and int(r['mode'])==curve['mode'] and int(r['repeat'])==point['repeat']]
            add('raw_median_'+str(trial)+'_'+curve['style']+'_'+str(curve['mode'])+'_'+str(point['repeat']),
                len(xs)==100 and abs(statistics.median(xs)-point['median_us'])<1e-12)
        # Independent least-squares calculation using raw medians for last two windows.
        byR={p['repeat']:p['median_us'] for p in curve['points']}
        for fit in curve['windows'][-2:]:
            x=fit['R']; y=[byR[r] for r in x]; n=len(x)
            slope=(n*sum(a*b for a,b in zip(x,y))-sum(x)*sum(y))/(n*sum(a*a for a in x)-sum(x)**2)
            add('slope_recalc_'+str(trial)+'_'+curve['style']+'_'+str(curve['mode'])+'_'+str(x[-1]),
                abs(slope-fit['slope_us_per_repeat'])<1e-10)

reports=sorted((root/'runs'/data['formal_tags']['ncu']).glob('*.ncu-rep'))
add('expected_ncu_reports',len(reports)==ncu_expected,len(reports))
report_hashes={}
for report in reports:
    p=subprocess.run(['/opt/nvidia/nsight-compute/2026.2.1/ncu','-i',str(report),'--csv','--page','raw'],capture_output=True,text=True,timeout=60)
    add('ncu_import_'+report.stem,p.returncode==0)
    report_hashes[report.name]=hashlib.sha256(report.read_bytes()).hexdigest()
    log=report.with_suffix('.log').read_text()
    add('no_sampling_overflow_'+report.stem,not any(word in log.lower() for word in ['overflow','dropped samples']))

core=[r for r in data['NCU'] if r['name'].startswith('profile_RUNTIME_')]
if ncu_expected==6:
    add('same_TMEM_MMA_TMA_dynamic_work',len(core)==3 and all(len({r['opcode_dynamic_account'][op]['Instructions Executed'] for r in core})==1 for op in ['LDTM','UTCHMMA','UTMALDG']))
else:
    old=json.loads((root/'analysis_primary_v1/RESULTS.json').read_text())
    old_core=next(r for r in old['NCU'] if r['name']=='profile_RUNTIME_M3_source')
    new_core=[r for r in data['NCU'] if r['name'].startswith('profile_')]
    add('same_TMEM_MMA_TMA_dynamic_work_followup',len(new_core)==1 and all(new_core[0]['opcode_dynamic_account'][op]['Instructions Executed']==old_core['opcode_dynamic_account'][op]['Instructions Executed'] for op in ['LDTM','UTCHMMA','UTMALDG']))
mic=[r for r in data['NCU'] if r['name'].startswith('micro_profile_')]
add('micro_no_MMA_TMA_local',len(mic)==(2 if ncu_expected==6 else 1) and all(r['opcode_dynamic_account'][op]['Instructions Executed']==0 for r in mic for op in ['UTCHMMA','UTMALDG','LDL','STL']))
repo=root.parents[1]/'upstreams/flashinfer'
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
add('upstream_pinned_and_clean',head=='e77a4a0d276367895c3b50a642fd8f326c03fb72' and dirty=='',{'HEAD':head,'status':dirty})
result={'passed':all(c['passed'] for c in checks),'checks':checks,'report_sha256':report_hashes,
        'sanitizer_status':'NOT_PASSED: application host crash before valid check; see raw failure log',
        'not_modified':['upstream','existing reports','cycle.xlsx','hardware assumptions']}
(root/('VALIDATION_'+sys.argv[1]+'.json')).write_text(json.dumps(result,indent=2))
print(json.dumps({'passed':result['passed'],'checks':len(checks),'failed':[c for c in checks if not c['passed']]},indent=2))
assert result['passed'],'Evidence validation failed'
