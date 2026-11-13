from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
O=Path(__file__).resolve().parent;V=O/'r4_graph_event_v2'
assert os.environ.get('K1_GPU_SESSION')
prep=json.loads((V/'CPU_PREPARATION.json').read_text())
assert hashlib.sha256((V/'run_probe_graph.py').read_bytes()).hexdigest()==prep['new_host_SHA256']
assert hashlib.sha256((V/'audited_event_graph.py').read_bytes()).hexdigest()==prep['graph_helper_SHA256']
out=O/'r4_graph_screen01';s=O/'r4_graph_screen01_supervision';s.mkdir(exist_ok=False)
cmd=[sys.executable,str(V/'run_probe_graph.py'),'--mode','screen','--points','128,256','--authorization',str(O/'R4_GRAPH_SCREEN_ADMISSION.json'),'--out',str(out)]
(s/'COMMAND.json').write_text(json.dumps(cmd,indent=2));start=time.time_ns();timed_out=False
with (s/'stdout').open('w') as stdout,(s/'stderr').open('w') as stderr:
    child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
    try:code=child.wait(timeout=120)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=5)
(s/'STATUS.json').write_text(json.dumps(dict(pid=child.pid,returncode=code,timed_out=timed_out,start_time_ns=start,end_time_ns=time.time_ns(),model_enabled=False),indent=2))
assert code==0 and not timed_out,(code,timed_out)
nodes=list((out/'GRAPH_NODES').glob('*.json'));assert len(nodes)==8
assert all(json.loads(p.read_text())['status']=='THREE_NODE_TWO_EDGE_EVENT_KERNEL_EVENT_PASS' for p in nodes)
subprocess.run([sys.executable,str(O/'analyze_r4_screen.py'),'--run',str(out)],check=True,timeout=120)
print('R4_GRAPH_SCREEN_COMPLETE_ALL8_GRAPHS_GATE_PASS',flush=True)
