from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
for phase,tag in [('slope','padded_slope_v2'),('ncu','padded_ncu_v1')]:
    subprocess.run([sys.executable,str(root/'run_padded_suite.py'),'--phase',phase,'--tag',tag],check=True)
