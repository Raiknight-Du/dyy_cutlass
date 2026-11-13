"""Exactly two admitted original R5 profile calls; no performance inference."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent;N=Path('/home/ubuntu/workspace/op12/k4_10h_native_r5_20261026');assert os.environ.get('K1_GPU_SESSION')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(N/'run_probe.py')=='561a3942b7f0793e1e2bdf2bc64d3903ae81386c26f25527c4b03a89cc423d40'
assert sha(N/'operand_source.py')=='5311b72c38a7c7efcbba7a15fb569bd1a28c924b04aa444e48df42f799eb3e0a'
assert sha(N/'analyze_ncu.py')=='87a17938e62e6751f2358b537674f0b7d39b7578d68c6a70d361abcd1cc111d1'
auth=O/'R5_NCU_ADMISSION.json';admission=json.loads(auth.read_text());assert admission['authorized_NCU_invocations']==2
out=O/'r5_ncu01';out.mkdir(exist_ok=False);metrics=json.loads((O/'r3_v2_ncu_S_G128_LOAD_01/METRICS.json').read_text());ncu='/opt/nvidia/nsight-compute/2026.2.1/ncu'
expected={'SS':'60b132f31202059cf0d7a29a49b2097f0c4109909de97a4645cf73706cd08038','TS':'34e6c4bdffb18e076cf1d0809a6645277a85c1e7e1a60ec47b92e1a1fb0b5ebf'}
for source in ['SS','TS']:
    config=source+'_LOAD';case=out/config;case.mkdir();report=case/config
    cmd=[ncu,'--target-processes','all','--replay-mode','application','--cache-control','none','--clock-control','none','--profile-from-start','off','--launch-count','1','--kernel-name','regex:.*operand_sourceOperandSourceProbe.*','--export',str(report),'--section','LaunchStats','--section','SourceCounters','--section','InstructionStats','--metrics',','.join(metrics),sys.executable,str(N/'run_probe.py'),'--mode','profile','--config',config,'--points','512','--authorization',str(auth),'--out',str(case/'replays')]
    cp=case/'COMMAND.json';cp.write_text(json.dumps(cmd,indent=2));row=dict(round='R5',configuration=config,start_time_ns=time.time_ns(),status='STARTED',model_enabled=False)
    with (O/'RUNNER_NCU_LEDGER.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    q=subprocess.run([sys.executable,str(O/'gpu_telemetry_sidecar.py'),'--out',str(case/'telemetry'),'--command-json',str(cp),'--interval-ms','100'],timeout=1800)
    row.update(end_time_ns=time.time_ns(),returncode=q.returncode,status='REPORT_CREATED' if q.returncode==0 and report.with_suffix('.ncu-rep').exists() else 'FAILED')
    (case/'RESULT.json').write_text(json.dumps(row,indent=2))
    with (O/'RUNNER_NCU_LEDGER.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    assert row['status']=='REPORT_CREATED',row
    for page in ['raw','source']:
        argv=[ncu,'--import',str(report.with_suffix('.ncu-rep')),'--page',page,'--csv','--print-units','base']
        if page=='source':argv+=['--print-source','sass']
        with (case/(config+'_'+page+'.csv')).open('x') as f:subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
    meta_files=list((case/'replays').glob('*/ACTUAL_METADATA_GATE_'+source+'.json'))
    assert meta_files
    for p in meta_files:
        m=json.loads(p.read_text());assert m['actual_CUBIN_SHA256']==expected[source] and m['CPU_text_exact_match'] and m['max_CTA_per_SM']==1 and m['helper_invoked']
    (case/'IDENTITY_RESOURCE_AUDIT.json').write_text(json.dumps(dict(status='ACTUAL_WHOLE_CUBIN_RESOURCE_PASS',CUBIN_SHA256=expected[source],metadata_replays=len(meta_files),clock_policy='automatic; counter evidence only; no replay timing adoption',model_enabled=False),indent=2))
    subprocess.run([sys.executable,str(N/'analyze_ncu.py'),'--configuration',source,'--source',str(case/(config+'_source.csv')),'--raw',str(case/(config+'_raw.csv')),'--out',str(case/'COUNTER_AUDIT.json')],check=True)
    audit=json.loads((case/'COUNTER_AUDIT.json').read_text());assert audit['status']=='DYNAMIC_MMA_AND_TC_COUNTS_PASS',audit['status']
    print(config,'WHOLE_CUBIN_AND_DYNAMIC_GATE_PASS',flush=True)
(out/'COMPLETE.json').write_text(json.dumps(dict(completed=True,valid_NCU=2,performance_samples_adopted=False,model_enabled=False),indent=2))
