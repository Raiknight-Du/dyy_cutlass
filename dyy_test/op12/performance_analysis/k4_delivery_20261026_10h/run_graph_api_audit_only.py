from pathlib import Path
import csv,hashlib,json,os,signal,subprocess,sys,time
O=Path(__file__).resolve().parent;V=O/'r4_graph_api_audit_only';assert os.environ.get('K1_GPU_SESSION')
for n,h in json.loads((V/'CPU_MANIFEST.json').read_text()).items():assert hashlib.sha256((V/n).read_bytes()).hexdigest()==h
out=O/'r4_graph_edge_api_diag01';sup=O/'r4_graph_edge_api_diag01_supervision';sup.mkdir(exist_ok=False)
cmd=[sys.executable,str(V/'run_audit_only.py'),'--mode','screen','--points','128','--authorization',str(O/'GRAPH_API_AUDIT_ONLY_ADMISSION.json'),'--out',str(out)]
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
assert not list(csv.DictReader((out/'EVENT_RAW.csv').open()))
assert not list(csv.DictReader((out/'CTA_CLOCK_RAW.csv').open()))
assert json.loads((out/'GRAPH_TIMING_PROTOCOL.json').read_text())['direct_lazy_invocations']==1
assert len(list((out/'GRAPH_NODES').glob('*.dot')))==1
record=json.loads((out/'GRAPH_NODES/N128_M0.json').read_text())
assert record['metadata']['graph_instantiates']==record['metadata']['graph_replays']==record['metadata']['timing_samples']==0
(out/'AUDIT_ONLY_COMPLETE.json').write_text(json.dumps(dict(status='API_QUERY_DIAGNOSTIC_COMPLETED_NOT_TIMING_ADMISSION',direct_lazy=1,captures=1,instantiates=0,replays=0,warmups=0,samples=0,model_enabled=False),indent=2))
print('API_QUERY_DIAGNOSTIC_COMPLETED_NOT_TIMING_ADMISSION',flush=True)
