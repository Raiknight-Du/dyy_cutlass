"""NCU application-replay launcher; Runner use only after Lead NCU admission.

Each replay application PID gets a fresh output directory. No CUDA context is
created by this Python launcher. Profiled child brackets only one N512 launch.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True);p.add_argument('--config',required=True)
    a=p.parse_args();gate=json.loads(a.gate.read_text());auth=json.loads(a.authorization.read_text())
    h=hashlib.sha256(a.binary.read_bytes()).hexdigest()
    assert gate['application_fatbin_native_only']
    assert h==gate['actual_executable_sha256']==auth['binary_sha256']
    assert auth['round']=='R3' and 'NCU' in auth['approved_modes'] and auth['max_N']>=512
    assert a.config in [f'{s}_G{g}_{m}' for s in ['S','V'] for g in [1,128] for m in ['LOAD','CONTROL']]
    a.out_root.mkdir(parents=True,exist_ok=True);out=a.out_root/f'application_{os.getpid()}'
    env=os.environ.copy();env.pop('CUDA_FORCE_PTX_JIT',None);env.pop('CUDA_DISABLE_PTX_JIT',None)
    cmd=[str(a.binary.resolve()),'--mode','profile','--out',str(out),'--config',a.config,
         '--points','512','--warmup','20','--samples','1']
    result=subprocess.run(cmd,env=env)
    (a.out_root/f'APPLICATION_{os.getpid()}.json').write_text(json.dumps({'command':cmd,
       'returncode':result.returncode,'binary_sha256':h,'model_enabled':False},indent=2))
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
