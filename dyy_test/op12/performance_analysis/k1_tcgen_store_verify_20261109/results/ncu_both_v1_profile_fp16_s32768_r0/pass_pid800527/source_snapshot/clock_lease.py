"""Only beneath the unified GPU guard; user authorized 1845 then automatic DVFS."""
import json,os,subprocess,sys,time
from pathlib import Path
out=Path(os.environ['K1_GUARD_DIR'])/'clock_lease';out.mkdir(exist_ok=False)
def capture(name,args,check=True):
    p=subprocess.run(args,text=True,capture_output=True)
    (out/(name+'.json')).write_text(json.dumps(dict(command=args,epoch=time.time(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2))
    if check and p.returncode:raise RuntimeError(name+' failed')
    return p
capture('before',['nvidia-smi','-q'])
try:
    capture('lock1845',['sudo','-n','nvidia-smi','-i','0','-lgc','1845,1845'])
    command=sys.argv[1:]
    p=subprocess.run(command,timeout=5400)
    code=p.returncode
finally:
    capture('restore_automatic',['sudo','-n','nvidia-smi','-i','0','-rgc'])
    capture('after',['nvidia-smi','-q'])
sys.exit(code)
