"""Explicit, bounded, single-GPU supervisor. Default is DRY RUN; no CUDA imports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

# Retain positive ancestry observations so NVML's short post-exit '[No data]'
# window cannot invalidate our own already-identified worker. A live PID is
# always checked anew; unobserved or unrelated PIDs remain external.
SEEN_OWN_CHILDREN = set()


def ancestor_owned(pid):
    original_pid = pid
    if pid in SEEN_OWN_CHILDREN and not Path(f'/proc/{pid}').exists():
        return True
    for _ in range(64):
        if pid == os.getpid():
            SEEN_OWN_CHILDREN.add(original_pid)
            return True
        if pid <= 1:
            return False
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def observe(gpu):
    # Snapshot descendants before querying NVML, which can outlive a replay
    # child. Only ancestry from our own supervisor is added to this set.
    pairs = subprocess.check_output(['ps','-e','-o','pid=,ppid='],text=True,timeout=10)
    parents = {int(v[0]):int(v[1]) for line in pairs.splitlines() if len(v:=line.split())==2}
    current = {os.getpid()}
    for _ in range(64):
        added = {p for p,parent in parents.items() if parent in current}-current
        if not added:break
        current.update(added)
    SEEN_OWN_CHILDREN.update(current)
    fields = 'uuid,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate'
    def query(args):
        return subprocess.check_output(['nvidia-smi', '-i', str(gpu), *args], text=True, timeout=10).strip()
    row = query([f'--query-gpu={fields}', '--format=csv,noheader,nounits'])
    apps = query(['--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'])
    values = [x.strip() for x in row.split(',')]
    if len(values) != 7:
        raise RuntimeError('Ambiguous GPU telemetry: ' + row)
    external = []
    for line in apps.splitlines():
        if not line.strip():
            continue
        pid = int(line.split(',')[0].strip())
        if not ancestor_owned(pid):
            external.append({'pid': pid, 'description': line})
    return {'utc_epoch': time.time(), 'gpu': dict(zip(fields.split(','), values)),
            'util': float(values[1]), 'external': external}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--max-wait', type=int, default=600)
    p.add_argument('--lock', type=Path, default=Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/.op12_gpu_runner.lock'))
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('A child command is required after --')
    if not a.execute:
        print(json.dumps({'status':'DRY_RUN_NO_GPU_QUERY', 'command':command, 'out':str(a.out)}, indent=2))
        return 0
    if sys.platform != 'linux':
        raise RuntimeError('GPU execution is only supported on the Linux B200 host')
    import fcntl
    a.out.mkdir(parents=True, exist_ok=False)
    a.lock.parent.mkdir(parents=True, exist_ok=True)
    records, invalid = [], []
    stop = threading.Event()
    status, returncode = 'WAITING_IDLE', None
    with a.lock.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            (a.out/'STATUS.json').write_text(json.dumps({'status':'BLOCKED_RUNNER_LOCK'}))
            return 3
        def save_observation():
            try:
                item = observe(a.gpu)
            except Exception as exc:
                item = {'utc_epoch':time.time(), 'error':repr(exc), 'external':[], 'util':None}
            records.append(item)
            with (a.out/'TELEMETRY.jsonl').open('a') as f:
                f.write(json.dumps(item)+'\n')
            return item
        try:
            checks, begin = 0, time.monotonic()
            while checks < 3:
                x = save_observation()
                checks = checks+1 if not x.get('error') and not x['external'] and x['util'] < 5 else 0
                print('IDLE_CHECK', checks, flush=True)
                if checks == 3:
                    break
                if time.monotonic()-begin+30 > a.max_wait:
                    status = 'BLOCKED_GPU_BUSY_OR_TELEMETRY'
                    return 3
                time.sleep(30)
            def monitor():
                while not stop.wait(1):
                    x = save_observation()
                    if x.get('error') or x['external']:
                        invalid.append(x)
                        (a.out/'INVALID_COMPETITION').touch()
            token = uuid.uuid4().hex
            env = os.environ.copy()
            env.update(K1_GPU_SESSION=token, K1_GUARD_DIR=str(a.out.resolve()),
                       CUDA_VISIBLE_DEVICES=str(a.gpu))
            # Do not change clocks, MPS, or external processes.
            worker = threading.Thread(target=monitor, daemon=True)
            worker.start()
            status = 'RUNNING'
            with (a.out/'command.log').open('w') as log:
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    returncode = proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()  # Only the child started by this guard.
                    proc.wait()
                    status = 'INTERRUPTED_OWN_CHILD'
                    raise
                finally:
                    stop.set()
                    worker.join(timeout=15)
            last = save_observation()
            if last.get('error') or last['external']:
                invalid.append(last)
            status = ('INVALID_COMPETITION' if invalid else
                      'COMPLETED_PENDING_REVIEW' if returncode == 0 else 'CHILD_FAILED')
            return 4 if invalid else (returncode or 0)
        finally:
            stop.set()
            (a.out/'STATUS.json').write_text(json.dumps({
                'status':status, 'command':command, 'returncode':returncode,
                'invalid_observations':invalid, 'automatic_clock_change':False,
                'model_enabled':False}, indent=2))


if __name__ == '__main__':
    sys.exit(main())
