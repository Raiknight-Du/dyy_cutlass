from pathlib import Path
import subprocess,sys
P=Path(__file__).parent
for S in (8192,16384):
 for trial in (1,2,3):
  target=P/f's{S}_r{trial}'
  with (P/f's{S}_r{trial}.log').open('w')as f:
   subprocess.run([sys.executable,str(P/'run_probe.py'),'--out',str(target),'--generated',str(P/'generated/k2_body_generated.py'),'--trial',str(trial),'--S',str(S)],stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',S,trial,flush=True)
