"""Bounded initial trace trial; further repetitions depend on offline audit."""
import json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
assert os.environ.get('K1_GPU_SESSION')
subprocess.run([sys.executable,str(R/'prepare_native_trace.py')],check=True)
subprocess.run([sys.executable,str(R/'prepare_runner.py')],check=True)
subprocess.run([sys.executable,str(R/'run_trace.py'),'--out',str(R/'trial_b32_t0'),'--trial','0','--sample-block','32','--warmup','20','--samples','100'],check=True,timeout=1200)
print('INITIAL_TRIAL_COMPLETE_REQUIRES_REVIEW',flush=True)
