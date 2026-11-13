from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
O=Path(__file__).resolve().parent;N=Path('/home/ubuntu/workspace/op12/k4_10h_native_r5_20261026');assert os.environ.get('K1_GPU_SESSION')
expected={'run_probe_graph.py':'8006953f28e2f8990e08760cbfc187ba01e48770313fe03b2bed283c0ff746cf','audited_event_graph.py':'f1c93643ebddd4338300e6b01dcdd13905c0a0273876632c4bbe07f9de787de4','graph_edge_native_audit.py':'2ab3981b328815531ad5cd73d445acecd266710a6935d5363f9e1e54619057e6','analyze_rates.py':'56db6c42d6f510cb22b82d7f108afb1a9031f2783ff4276eddab89d5fdcbe63c'}
for name,h in expected.items():assert hashlib.sha256((N/name).read_bytes()).hexdigest()==h,(name,hashlib.sha256((N/name).read_bytes()).hexdigest())
for source in ['SS','TS']:
    d=json.loads((O/'r5_ncu01'/(source+'_LOAD')/'COUNTER_AUDIT.json').read_text());assert d['status']=='DYNAMIC_MMA_AND_TC_COUNTS_PASS'
out=O/'r5_graph_formal_ext16384_01';out.mkdir(exist_ok=False);records=[]
for trial in range(3):
    cmd=[sys.executable,str(N/'run_probe_graph.py'),'--mode','formal','--trial',str(trial),'--points','128,256,512,1024,2048,4096,8192,16384','--authorization',str(O/'R5_GRAPH_FORMAL_EXT_ADMISSION.json'),'--out',str(out/f'trial{trial}')]
    started=time.time_ns();timed_out=False
    with (out/f'trial{trial}.stdout').open('w') as stdout,(out/f'trial{trial}.stderr').open('w') as stderr:
        child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
        try:code=child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=5)
    records.append(dict(trial=trial,pid=child.pid,command=cmd,start_time_ns=started,end_time_ns=time.time_ns(),returncode=code,timed_out=timed_out,model_enabled=False))
    (out/'RUN_RECORD.json').write_text(json.dumps(records,indent=2));assert code==0 and not timed_out,records[-1]
    nodes=list((out/f'trial{trial}'/'GRAPH_NODES').glob('N*.json'));assert len(nodes)==32
    assert all(json.loads(p.read_text())['status']=='THREE_NODE_TWO_EDGE_EVENT_KERNEL_EVENT_PASS' for p in nodes)
subprocess.run([sys.executable,str(N/'analyze_rates.py'),'--input',str(out),'--out',str(out/'RATE_ANALYSIS.json')],check=True,timeout=120)
