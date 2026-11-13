"""CPU-only native application build; no CUDA context or kernel launch."""
from pathlib import Path
import subprocess,json,hashlib
O=Path(__file__).resolve().parent;B=O/'compile01_cuda130';B.mkdir(exist_ok=False)
nvcc='/usr/local/cuda-13.0/bin/nvcc';cuobj='/usr/local/cuda-13.0/bin/cuobjdump'
commands=[]
def run(argv,name):
    commands.append(argv)
    q=subprocess.run(argv,cwd=B,text=True,capture_output=True)
    (B/(name+'.stdout')).write_text(q.stdout);(B/(name+'.stderr')).write_text(q.stderr)
    assert q.returncode==0,(name,q.returncode,q.stderr)
    return q.stdout
run([nvcc,'--version'],'nvcc_version')
run([nvcc,'-std=c++17','-gencode','arch=compute_100a,code=sm_100a','-O3','-lineinfo','-Xptxas=-v','--keep','--keep-dir','.',str(O/'ldg_service.cu'),'-o','ldg_service'],'build')
run([nvcc,'-std=c++17','-arch=compute_100a','-O3','-lineinfo','-ptx',str(O/'ldg_service.cu'),'-o','audit.compute_100a.ptx'],'audit_ptx')
for flags,name in [(['--dump-resource-usage'],'actual_resource'),(['--dump-sass'],'actual_executable'),(['--list-elf'],'native_elf_list'),(['--list-ptx'],'ptx_list'),(['--extract-elf','all'],'extract')]:
    result=run([cuobj,*flags,'ldg_service'],name)
    if name=='actual_executable':(B/'actual_executable.sass').write_text(result)
    if name=='actual_resource':(B/'actual_resource.txt').write_text(result)
assert not (B/'ptx_list.stdout').read_text().strip(),'Application executable unexpectedly embeds PTX'
(B/'COMMANDS.json').write_text(json.dumps(commands,indent=2))
(B/'HASHES.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in B.iterdir() if p.is_file()},indent=2))
print('CPU_BUILD_COMPLETE_NO_GPU',flush=True)
