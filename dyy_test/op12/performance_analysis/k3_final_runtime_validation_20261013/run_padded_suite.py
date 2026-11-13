"""Exclusive serial runner with frequency ledger and guaranteed clock reset."""
import argparse
import collections
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import pynvml as nv
ROOT = Path(__file__).resolve().parent
PY = ROOT.parents[1] / '.venv/bin/python'
NCU = '/opt/nvidia/nsight-compute/2026.2.1/ncu'
UUID = 'GPU-6abbea3c-b8d3-7f4b-ca76-85d612930fdf'

def cmd(argv):
    p = subprocess.run(list(map(str, argv)), text=True, capture_output=True, timeout=30)
    return {'argv': list(map(str, argv)), 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr, 'unix_ns': time.time_ns()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['smoke', 'event', 'slope', 'ncu', 'sanitizer'], required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--repeats', nargs='+', default=['16', '32', '64', '128', '256', '512', '1024', '2048'])
    a = p.parse_args()
    out = ROOT / 'runs' / a.tag
    out.mkdir(parents=True, exist_ok=False)
    ledger = {'phase': a.phase, 'idle_checks': [], 'commands': [], 'prior_lock_range': 'UNKNOWN; reset to driver defaults'}
    for i in range(3):
        apps = cmd(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
        status = cmd(['nvidia-smi', '--query-gpu=uuid,utilization.gpu,clocks.sm', '--format=csv,noheader'])
        ledger['idle_checks'].append({'apps': apps, 'status': status})
        assert apps['returncode'] == 0 and (not apps['stdout'].strip()), 'External process, no tests launched'
        assert UUID in status['stdout'] and ', 0 %,' in status['stdout'], 'GPU not idle'
        print(json.dumps({'idle_check': i + 1}), flush=True)
        if i < 2:
            time.sleep(30)
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByUUID(UUID)
    stop = threading.Event()
    pg = [None]
    foreign = []
    hist = collections.Counter()
    child = None
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGHUP, lambda *_: stop.set())

    def monitor():
        with (out / 'frequency.jsonl').open('x') as f:
            last = 0
            while not stop.is_set():
                now = time.time_ns()
                r = {'unix_ns': now, 'sm_mhz': nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM), 'memory_mhz': nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM), 'temperature_C': nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU), 'power_W': nv.nvmlDeviceGetPowerUsage(h) / 1000}
                hist[r['sm_mhz']] += 1
                if now - last > 500000000:
                    last = now
                    for proc in nv.nvmlDeviceGetComputeRunningProcesses(h):
                        try:
                            if os.getpgid(proc.pid) != pg[0]:
                                foreign.append({'unix_ns': now, 'pid': proc.pid})
                        except ProcessLookupError:
                            pass
                f.write(json.dumps(r) + '\n')
                f.flush()
                stop.wait(0.01)

    def run(argv, name, timeout=720):
        nonlocal child
        assert not stop.is_set() and (not foreign), 'Competition invalidates sample'
        start = time.time_ns()
        print(json.dumps({'start': name}), flush=True)
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['CUDA_VISIBLE_DEVICES'] = UUID
        with (out / (name + '.log')).open('x') as f:
            child = subprocess.Popen(list(map(str, argv)), stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            pg[0] = child.pid
            while child.poll() is None:
                if stop.is_set() or foreign or time.time_ns() - start > timeout * 1000000000.0:
                    os.killpg(child.pid, signal.SIGTERM)
                    raise RuntimeError('owned child stopped: deadline/competition/user stop')
                time.sleep(0.2)
            code = child.returncode
            pg[0] = None
        ledger['commands'].append({'name': name, 'argv': list(map(str, argv)), 'start_ns': start, 'end_ns': time.time_ns(), 'returncode': code})
        print(json.dumps({'completed': name, 'returncode': code}), flush=True)
        assert code == 0, name + ' failed; inspect log'
    th = None
    locked = False
    try:
        locked = True
        ledger['lock'] = cmd(['sudo', '-n', 'nvidia-smi', '-i', UUID, '--lock-gpu-clocks=1845,1845', '--mode=0'])
        assert ledger['lock']['returncode'] == 0
        time.sleep(0.25)
        assert nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM) == 1845
        th = threading.Thread(target=monitor, daemon=True)
        th.start()
        if a.phase == 'sanitizer':
            for phase in ['profile', 'micro_profile']:
                run(['/usr/local/cuda/bin/compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '7', PY, ROOT / 'run_padded.py', '--phase', phase, '--style', 'COALESCED', '--mode', '3', '--out', out / phase / 'host'], phase, timeout=300)
        elif a.phase == 'ncu':
            jobs = [('profile', 'PADDED', 3), ('micro_profile', 'PADDED', 3)]
            for (phase, style, mode) in jobs:
                name = phase + '_' + style + '_M' + str(mode)
                run([NCU, '--target-processes', 'all', '--nvtx', '--nvtx-include', 'K3_RUNTIME_TARGET/', '--replay-mode', 'application', '--cache-control', 'none', '--clock-control', 'none', '--warp-sampling-interval', '0', '--section', 'LaunchStats', '--section', 'SchedulerStats', '--section', 'WarpStateStats', '--section', 'MemoryWorkloadAnalysis', '--section', 'ComputeWorkloadAnalysis', '--section', 'SourceCounters', '--section', 'InstructionStats', '-o', out / name, PY, ROOT / 'run_padded.py', '--phase', phase, '--style', style, '--mode', mode, '--out', out / name / 'host'], name, timeout=900)
                for (page, extra) in [('raw', []), ('source', ['--print-source', 'sass'])]:
                    with (out / (name + '_' + page + '.csv')).open('x') as f:
                        result = subprocess.run([NCU, '-i', str(out / name) + '.ncu-rep', '--csv', '--page', page, *extra], stdout=f, stderr=subprocess.PIPE, text=True, timeout=120)
                        assert result.returncode == 0, result.stderr
        else:
            for trial in range(1, 2 if a.phase == 'smoke' else 4):
                name = a.phase + '_t' + str(trial)
                run([PY, ROOT / 'run_padded.py', '--phase', a.phase, '--trial', trial, '--out', out / name, '--repeats', *a.repeats], name, timeout=1200)
        ledger['status'] = 'COMPLETED'
    except BaseException as e:
        ledger['status'] = 'FAILED_OR_PAUSED'
        ledger['error'] = repr(e)
        raise
    finally:
        stop.set()
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        if th:
            th.join(timeout=5)
        if locked:
            ledger['reset'] = cmd(['sudo', '-n', 'nvidia-smi', '-i', UUID, '--reset-gpu-clocks'])
        ledger['sm_mhz_histogram'] = dict(hist)
        ledger['competition'] = foreign
        ledger['frequency_valid'] = bool(hist) and set(hist) == {1845} and (not foreign)
        with (out / 'RUN_LEDGER.json').open('x') as f:
            json.dump(ledger, f, indent=2)
        nv.nvmlShutdown()
        print(json.dumps({'status': ledger['status'], 'frequency_valid': ledger['frequency_valid']}), flush=True)
if __name__ == '__main__':
    main()
