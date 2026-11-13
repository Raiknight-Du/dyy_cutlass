"""Sequential safety and counter evidence; no GPU concurrency."""
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
for phase in ['sanitizer','ncu']:
    subprocess.run([sys.executable,str(root/'run_suite.py'),'--phase',phase,'--tag',phase+'_v1'],check=True)
