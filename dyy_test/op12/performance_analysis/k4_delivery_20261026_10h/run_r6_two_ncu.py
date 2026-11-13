"""Exactly two original full-CP counter captures; old application untouched."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent;old=Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/k4_init_layout_matched_20261024_2_run01')
assert os.environ.get('K1_GPU_SESSION')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old/'run_matched.py')=='82889a50c9cb6edddf8d23efa491a215afc9470f59c8836b5fabd3c700d0b2d4'
assert sha(old/'APPROVAL.json')=='e0cef34783a432f691e62b49ea1c69f2a9e05f43848778b5c2bae3b1e44e79bc'
expected={'B0':'06ebd78963371a30adac142c06b86950ec300b6cb762a209c5afc67628941814','V0':'6ca13a1d315d7a36ca221e0ec1752dfb1d6c2c4ed2acb6a574db925ed0fce6f4'}
out=O/'r6_ncu01';out.mkdir(exist_ok=False);env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
(out/'ADMISSION.json').write_text(json.dumps(dict(round='R6',approved_NCU_invocations=2,original_script=str(old/'run_matched.py'),source_SHA256=sha(old/'run_matched.py'),expected_CUBIN=expected,PYTHONDONTWRITEBYTECODE='1',full_CP_history=True,new_performance_or_kernel=False,prediction_only_Data_read_difference=1376256,model_enabled=False),indent=2))
ncu='/opt/nvidia/nsight-compute/2026.2.1/ncu';metrics=json.loads((O/'r3_v2_ncu_S_G128_LOAD_01/METRICS.json').read_text())
for role in ['B0','V0']:
    case=out/role;case.mkdir();report=case/role
    cmd=[ncu,'--replay-mode','application','--cache-control','none','--clock-control','none','--target-processes','all','--nvtx','--nvtx-include','K4_MATCHED/','--kernel-name','regex:.*CPDeltaRulePrefillTcgen05Sm100.*','--launch-count','1','--export',str(report),'--section','LaunchStats','--section','SourceCounters','--section','InstructionStats','--metrics',','.join(metrics),sys.executable,str(old/'run_matched.py'),'--mode','profile','--profile',role,'--out',str(case/'replays')]
    cp=case/'COMMAND.json';cp.write_text(json.dumps(cmd,indent=2))
    row=dict(round='R6',configuration=role,start_time_ns=time.time_ns(),status='STARTED',model_enabled=False)
    with (O/'RUNNER_NCU_LEDGER.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    q=subprocess.run([sys.executable,str(O/'gpu_telemetry_sidecar.py'),'--out',str(case/'telemetry'),'--command-json',str(cp),'--interval-ms','100'],env=env,timeout=1800)
    row.update(returncode=q.returncode,end_time_ns=time.time_ns(),status='REPORT_CREATED' if q.returncode==0 and report.with_suffix('.ncu-rep').exists() else 'FAILED')
    (case/'RESULT.json').write_text(json.dumps(row,indent=2))
    with (O/'RUNNER_NCU_LEDGER.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    assert row['status']=='REPORT_CREATED',row
    for page in ['raw','source']:
        argv=[ncu,'--import',str(report.with_suffix('.ncu-rep')),'--page',page,'--csv','--print-units','base']
        if page=='source':argv+=['--print-source','sass']
        with (case/(role+'_'+page+'.csv')).open('x') as f:subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
    evidence=[]
    for f in (case/'replays').glob('*/PROFILE_COMPLETE.json'):
        d=json.loads(f.read_text());assert d['role']==role and d['hash']==expected[role] and d['full_CP'] and d['cold_before_K1'];evidence.append(str(f.relative_to(case)))
    assert evidence,'No successful original application profile evidence'
    (case/'IDENTITY_HISTORY_AUDIT.json').write_text(json.dumps(dict(status='ORIGINAL_FULL_CP_EXACT_CUBIN_PASS',PROFILE_COMPLETE_files=evidence,original_script_unchanged=sha(old/'run_matched.py'),old_APPROVAL_unchanged=sha(old/'APPROVAL.json'),model_enabled=False),indent=2))
    print('R6',role,'NCU_COMPLETE',flush=True)
(out/'COMPLETE.json').write_text(json.dumps(dict(completed=True,formal_NCU_invocations=2,new_kernel=False,new_performance=False,model_enabled=False),indent=2))
