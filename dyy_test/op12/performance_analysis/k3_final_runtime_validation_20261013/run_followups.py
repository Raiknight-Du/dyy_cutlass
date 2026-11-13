"""Run only after the initial NCU owner has exited."""
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
jobs=[('run_batch_suite.py','event','event_batch64_v1'),
      ('run_layout_suite.py','event','layout_event_v1'),
      ('run_layout_suite.py','slope','layout_slope_v1'),
      ('run_layout_suite.py','ncu','layout_ncu_v1')]
for script,phase,tag in jobs:
    subprocess.run([sys.executable,str(root/script),'--phase',phase,'--tag',tag],check=True)
