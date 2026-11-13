from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
O=Path(__file__).resolve().parent;N=Path('/home/ubuntu/workspace/op12/k4_10h_native_r5_20261026');assert os.environ.get('K1_GPU_SESSION')
expected={'run_probe_graph.py':'8006953f28e2f8990e08760cbfc187ba01e48770313fe03b2bed283c0ff746cf','audited_event_graph.py':'f1c93643ebddd4338300e6b01dcdd13905c0a0273876632c4bbe07f9de787de4','graph_edge_native_audit.py':'2ab3981b328815531ad5cd73d445acecd266710a6935d5363f9e1e54619057e6'}
for name,h in expected.items():assert hashlib.sha256((N/name).read_bytes()).hexdigest()==h
out=O/'r5_graph_screen01';sup=O/'r5_graph_screen01_supervision';sup.mkdir(exist_ok=False)
cmd=[sys.executable,str(N/'run_probe_graph.py'),'--mode','screen','--points','128,256','--authorization',str(O/'R5_GRAPH_SCREEN_ADMISSION.json'),'--out',str(out)]
(sup/'COMMAND.json').write_text(json.dumps(cmd,indent=2));start=time.time_ns();timed_out=False
with (sup/'stdout').open('w') as stdout,(sup/'stderr').open('w') as stderr:
    child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
    try:code=child.wait(timeout=120)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=5)
(sup/'STATUS.json').write_text(json.dumps(dict(pid=child.pid,returncode=code,timed_out=timed_out,start_time_ns=start,end_time_ns=time.time_ns(),model_enabled=False),indent=2))
assert code==0 and not timed_out,(code,timed_out)
done=json.loads((out/'completion.json').read_text());assert done['completed'] and done['correctness'] and done['invocations']==40
nodes=list((out/'GRAPH_NODES').glob('N*.json'));assert len(nodes)==8
assert all(json.loads(p.read_text())['status']=='THREE_NODE_TWO_EDGE_EVENT_KERNEL_EVENT_PASS' for p in nodes)
print('R5_GRAPH_SCREEN40CALLS_8NODECERTS_COMPLETE',flush=True)
