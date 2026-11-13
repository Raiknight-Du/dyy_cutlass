"""Lead-approved finite screen. No formal or NCU launch in this script."""
from pathlib import Path
import json,os,subprocess,sys
O=Path(__file__).resolve().parent;V=O/'r3_ldg_service_v2'
assert os.environ.get('K1_GPU_SESSION')
command=[sys.executable,str(V/'run_batch.py'),'--binary',str(V/'compile01_cuda130/ldg_service'),
 '--gate',str(V/'STATIC_GATE.json'),'--authorization',str(O/'R3_V2_SCREEN_NCU_ADMISSION.json'),
 '--out',str(O/'r3_v2_screen01'),'--mode','screen','--processes','1','--points','128,256']
p=O/'R3_V2_SCREEN_COMMAND.json';p.write_text(json.dumps(command,indent=2))
subprocess.run([sys.executable,str(O/'gpu_telemetry_sidecar.py'),'--out',str(O/'r3_v2_screen01_telemetry'),
 '--command-json',str(p),'--interval-ms','100'],check=True,timeout=600)
subprocess.run([sys.executable,str(O/'analyze_r3_screen.py'),'--run',str(O/'r3_v2_screen01')],check=True,timeout=120)
