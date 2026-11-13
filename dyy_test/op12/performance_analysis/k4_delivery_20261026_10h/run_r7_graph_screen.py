from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
O=Path(__file__).resolve().parent;N=Path('/home/ubuntu/workspace/op12/k4_10h_native_r7_20261026');assert os.environ.get('K1_GPU_SESSION')
for name,h in {'run_probe_graph.py':'3c7127fa511927d28ed2eb06712958ed63ab258c2d8dbb3a4c612a9715abbb96','audited_event_graph.py':'f1c93643ebddd4338300e6b01dcdd13905c0a0273876632c4bbe07f9de787de4','graph_edge_native_audit.py':'2ab3981b328815531ad5cd73d445acecd266710a6935d5363f9e1e54619057e6'}.items():assert hashlib.sha256((N/name).read_bytes()).hexdigest()==h,name
out=O/'r7_graph_screen01';sup=O/'r7_graph_screen01_supervision';sup.mkdir(exist_ok=False)
cmd=[sys.executable,str(N/'run_probe_graph.py'),'--mode','screen','--points','128,256','--authorization',str(O/'R7_GRAPH_SCREEN_ADMISSION.json'),'--out',str(out)]
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
done=json.loads((out/'completion.json').read_text());assert done['completed'] and done['correctness'] and done['invocations']==20
nodes=list((out/'GRAPH_NODES').glob('N*.json'));assert len(nodes)==4
assert all(json.loads(p.read_text())['status']=='THREE_NODE_TWO_EDGE_EVENT_KERNEL_EVENT_PASS' for p in nodes)
print('R7_FINITE_SCREEN20CALLS_4NODECERTS_COMPLETE',flush=True)
