"""Exactly two Lead-approved counter-scope diagnostics; unchanged R3 application."""
from pathlib import Path
import csv,hashlib,json,os,signal,subprocess,sys,time
O=Path(__file__).resolve().parent;N=O/'r3_ldg_service_v2'
assert os.environ.get('K1_GPU_SESSION')
assert not (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists()
auth=O/'R9_NCU_ADMISSION.json';a=json.loads(auth.read_text())
assert a['diagnostic_round']=='R9' and a['authorized_invocations']==2
assert a['allowed_configurations']==['S_G128_LOAD'] and a['max_N']==512
binary=N/'compile01_cuda130/ldg_service'
for p,h in [(binary,a['binary_sha256']),(N/'profile_one.py','93716a553a998e42c2c2183d80c59505ddcbb41e302de56475d29e2b3eab7179'),(N/'ldg_service.cu','49bf11c98c6eb0d3812f57d90c70a36783f4d18cbbeeb645b24a6fcb900fec7f'),(N/'STATIC_GATE.json','0174a3b75f7a4f725f734b34e101c0786335983445ee2c4a0e555ddbb65c1d74')]:
    assert hashlib.sha256(p.read_bytes()).hexdigest()==h,str(p)
metrics=['dram__bytes_read.sum','lts__t_sectors_srcunit_tex_op_read.sum','lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum','lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum','l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_read.sum','sm__cycles_elapsed.avg','l1tex__cycles_elapsed.avg','lts__cycles_elapsed.avg','dram__cycles_elapsed.avg','gpu__time_duration.sum','profiler__replayer_passes']
ncu='/opt/nvidia/nsight-compute/2026.2.1/ncu';out=O/'r9_hw_scope01';out.mkdir(exist_ok=False)
# Catalogue query only; chip-name query does not launch a target.
qcmd=[ncu,'--query-metrics','--chips','gb100','--query-metrics-mode','suffix','--metrics','l1tex__cycles_elapsed','--csv']
q=subprocess.run(qcmd,capture_output=True,text=True,check=True,timeout=120)
(out/'L1_CLOCK_QUERY.csv').write_text(q.stdout)
assert 'l1tex__cycles_elapsed.avg' in {r['Metric name'] for r in csv.DictReader(q.stdout.splitlines())}
(out/'METRICS.json').write_text(json.dumps(metrics,indent=2));(out/'QUERY_COMMAND.json').write_text(json.dumps(qcmd,indent=2))
records=[]
for i in range(2):
    d=out/f'invocation{i+1}';d.mkdir()
    cmd=[ncu,'--target-processes','all','--replay-mode','application','--cache-control','none','--clock-control','none','--profile-from-start','off','--launch-count','1','--export',str(d/'S_G128_LOAD'),'--metrics',','.join(metrics),sys.executable,str(N/'profile_one.py'),'--binary',str(binary),'--gate',str(N/'STATIC_GATE.json'),'--authorization',str(auth),'--config','S_G128_LOAD','--out-root',str(d/'replays')]
    assert '--section' not in cmd and '--set' not in cmd
    (d/'COMMAND.json').write_text(json.dumps(cmd,indent=2));start=time.time_ns()
    sup=[sys.executable,str(O/'gpu_telemetry_sidecar.py'),'--out',str(d/'telemetry'),'--command-json',str(d/'COMMAND.json'),'--interval-ms','100']
    timed_out=False;child=subprocess.Popen(sup,start_new_session=True)
    try:code=child.wait(timeout=600)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=5)
    record=dict(round='R9',configuration='S_G128_LOAD',N=512,invocation=i+1,kind='COUNTER_SCOPE_DIAGNOSTIC',start_time_ns=start,end_time_ns=time.time_ns(),returncode=code,supervisor_PID=child.pid,timed_out=timed_out,clock_policy='temporary1845request_actual_sampled',model_enabled=False)
    (d/'RESULT.json').write_text(json.dumps(record,indent=2));records.append(record)
    (out/'RUN_RECORD.json').write_text(json.dumps(records,indent=2))
    assert code==0 and not timed_out,record
    report=d/'S_G128_LOAD.ncu-rep';assert report.exists()
    with (d/'S_G128_LOAD_raw.csv').open('x') as f:subprocess.run([ncu,'--import',str(report),'--page','raw','--csv','--print-units','base'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
    done=[json.loads(p.read_text()) for p in (d/'replays').rglob('completion.json')]
    maps=[json.loads(p.read_text()) for p in (d/'replays').rglob('ADDRESS_MAPPING.json')]
    assert done and len(done)==len(maps)
    assert all(x['completed'] and x['correctness'] and x['invocations']==1 and x['warmups_per_config_per_point']==20 and x['samples_per_point']==1 for x in done)
    assert all(x['allocation_B']==67108864 and x['mapping']=='slot-major' for x in maps)
    rows=list(csv.DictReader((d/'S_G128_LOAD_raw.csv').open()));r=rows[-1]
    assert float(r['l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_read.sum'].replace(',',''))==1073741824
    vals={k:r.get(k,'UNKNOWN') for k in metrics}
    (d/'COUNTER_SCOPE_AUDIT.json').write_text(json.dumps(dict(status='UNCHANGED_APPLICATION_HISTORY_AND_DATA_WF_PASS',replay_applications=len(done),counters=vals,actual_passes=r.get('profiler__replayer_passes','UNKNOWN'),model_enabled=False),indent=2))
(out/'COMPLETE.json').write_text(json.dumps(dict(completed=True,NCU_invocations=2,sections=[],SourceCounters=False,InstructionStats=False,no_more_GPU=True,model_enabled=False),indent=2))
print('R9_TWO_COUNTER_SCOPE_DIAGNOSTICS_COMPLETE_STOP_GPU',flush=True)
