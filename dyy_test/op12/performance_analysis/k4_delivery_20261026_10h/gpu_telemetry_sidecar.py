"""Read-only NVIDIA telemetry, optionally supervising an unmodified command.

Creates no CUDA context, changes no clocks, and never infers missing reasons.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--command-json',type=Path);p.add_argument('--interval-ms',type=int,default=100);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
helptext=subprocess.check_output(['nvidia-smi','--help-query-gpu'],text=True)
(a.out/'NVIDIA_SMI_SUPPORTED_FIELDS.txt').write_text(helptext)
supported=set(re.findall(r'"([a-zA-Z0-9_.]+)"',helptext))
requested=['uuid','pstate','utilization.gpu','utilization.memory','clocks.sm','clocks.mem','temperature.gpu','temperature.memory',
 'power.draw','power.draw.instant','power.limit','enforced.power.limit']
requested += ['clocks_event_reasons.'+x for x in ['supported','active','gpu_idle','applications_clocks_setting','sw_power_cap','hw_slowdown','hw_thermal_slowdown','hw_power_brake_slowdown','sw_thermal_slowdown','sync_boost','display_clock_setting']]
requested += ['clocks_event_reasons_counters.'+x for x in ['sw_power_cap','sync_boost','sw_thermal_slowdown','hw_thermal_slowdown','hw_power_brake_slowdown']]
fields=[x for x in requested if x in supported]
missing=[x for x in requested if x not in supported]
command=['nvidia-smi','-i','0','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits']
manifest=dict(requested=requested,selected=fields,unsupported={x:'UNKNOWN' for x in missing},query_command=command,
 query_help_sha256=hashlib.sha256(helptext.encode()).hexdigest(),interval_ms=a.interval_ms,
 scope='Whole supervised command; snapshots are bounded by host read start/end, not attributed to individual CUDA kernel endpoints',clock_management=False,model_enabled=False)
(a.out/'TELEMETRY_SCHEMA.json').write_text(json.dumps(manifest,indent=2))
def snapshot(phase):
    start=time.time_ns();q=subprocess.run(command,text=True,capture_output=True,timeout=10);end=time.time_ns()
    values={x:'UNKNOWN' for x in requested};raw={};error=None
    if q.returncode==0:
        rows=list(csv.reader(q.stdout.splitlines()))
        if len(rows)==1 and len(rows[0])==len(fields):
            raw=dict(zip(fields,[x.strip() for x in rows[0]]))
            values.update({k:('UNKNOWN' if v in ['N/A','[N/A]','Not Supported','[Not Supported]'] else v) for k,v in raw.items()})
        else:error='Unexpected query row/column count'
    else:error=q.stderr.strip() or q.stdout.strip()
    record=dict(phase=phase,read_start_time_ns=start,read_end_time_ns=end,values=values,raw=raw,returncode=q.returncode,error=error)
    with (a.out/'SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    return record

before=snapshot('BEFORE_COMMAND' if a.command_json else 'POSTSCREEN_CURRENT_STATE_ONLY')
(a.out/'BEFORE.json').write_text(json.dumps(before,indent=2))
if not a.command_json:
    (a.out/'LIMITATION.json').write_text(json.dumps(dict(screen_in_run_reason_data='UNKNOWN',reason='Telemetry added after screen; current values cannot be retroactively assigned to screen kernels',model_enabled=False),indent=2))
    print(json.dumps(dict(status='READ_ONLY_CURRENT_STATE_CAPTURED',selected=len(fields),unsupported=missing,query_returncode=before['returncode'])));raise SystemExit(0)
argv=json.loads(a.command_json.read_text());assert isinstance(argv,list) and all(isinstance(x,str) for x in argv)
(a.out/'SUPERVISED_COMMAND.json').write_text(json.dumps(argv,indent=2))
stop=threading.Event()
def monitor():
    while not stop.is_set():
        try:snapshot('COMMAND_RUNNING')
        except Exception as exc:
            with (a.out/'SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps(dict(phase='COMMAND_RUNNING',time_ns=time.time_ns(),values='UNKNOWN',error=repr(exc)))+'\n')
        stop.wait(a.interval_ms/1000)
thread=threading.Thread(target=monitor,daemon=True)
with (a.out/'COMMAND.stdout').open('x') as stdout,(a.out/'COMMAND.stderr').open('x') as stderr:
    proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr);thread.start()
    try:code=proc.wait()
    finally:stop.set();thread.join(timeout=12)
after=snapshot('AFTER_COMMAND')
(a.out/'AFTER.json').write_text(json.dumps(after,indent=2))
(a.out/'STATUS.json').write_text(json.dumps(dict(returncode=code,pid=proc.pid,clock_management=False,model_enabled=False),indent=2))
print(json.dumps(dict(status='SUPERVISED_COMMAND_COMPLETE',returncode=code,model_enabled=False)))
raise SystemExit(code)
