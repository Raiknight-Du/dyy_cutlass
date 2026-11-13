"""Lead-submitted JSON jobs, serialized beneath the idle/competition guard."""
import json, os, subprocess, time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
jobs=root/'jobs'; jobs.mkdir(exist_ok=True)
deadline=time.monotonic()+5400
while time.monotonic()<deadline:
    if (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists():
        raise RuntimeError('GPU competition or failed telemetry: stop further jobs')
    if (jobs/'STOP').exists():break
    for f in sorted(jobs.glob('*.json')):
        done=f.with_suffix('.done.json')
        if f.name.endswith('.done.json') or done.exists():continue
        try: job=json.loads(f.read_text())
        except json.JSONDecodeError: continue
        cmd=job['command']
        print('JOB_START',f.name,cmd,flush=True)
        start=time.time()
        with f.with_suffix('.log').open('w') as log:
            p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=job.get('timeout',1200))
        done.write_text(json.dumps(dict(command=cmd,returncode=p.returncode,epoch_start=start,epoch_end=time.time()),indent=2))
        print('JOB_END',f.name,p.returncode,flush=True)
    time.sleep(1)
print('QUEUE_FINISHED',flush=True)
