"""Confirm admitted coarse design only; no additional point/storage revisions."""
import os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
assert os.environ.get('K1_GPU_SESSION')
for block,trial in [(32,1),(32,2),(0,0),(63,0)]:
    print('COARSE_CONFIRM',block,trial,flush=True)
    subprocess.run([sys.executable,str(R/'run_trace.py'),'--out',str(R/f'coarse_b{block}_t{trial}'),'--trial',str(trial),'--sample-block',str(block),'--variants','B0,B1,COARSE','--warmup','20','--samples','100'],check=True,timeout=1200)
print('COARSE_CONFIRM_COMPLETE',flush=True)
