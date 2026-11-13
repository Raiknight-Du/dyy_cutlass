"""Offline export of already collected NCU reports; no kernel launch."""
import subprocess,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
ncu='/opt/nvidia/nsight-compute/2026.2.1/ncu'
for report in sorted((root/'ncu').glob('*.ncu-rep')):
    cmd=[ncu,'--import',str(report),'--page','raw','--csv']
    result=subprocess.run(cmd,capture_output=True,text=True,check=True)
    report.with_suffix('.raw.csv').write_text(result.stdout)
    report.with_suffix('.export.json').write_text(json.dumps(dict(command=cmd,returncode=result.returncode,stderr=result.stderr),indent=2))
    print('EXPORTED',report.name,flush=True)
