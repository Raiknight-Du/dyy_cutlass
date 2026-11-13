"""Driver occupancy query for exact saved CUBIN; no kernel launch or timing."""
import argparse
import json
import subprocess
from pathlib import Path
from cuda.bindings import driver as cu

p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--output',required=True);a=p.parse_args()
def check(result):
    if int(result[0])!=0: raise RuntimeError(str(result))
    return result[1:]
check(cu.cuInit(0))
device,=check(cu.cuDeviceGet(0));context,=check(cu.cuDevicePrimaryCtxRetain(device));check(cu.cuCtxSetCurrent(context))
records=[]
for file in Path(a.binary).glob('*.cubin'):
    k='K1' if 'TPrecompute' in file.name else 'K2' if 'MNPrecompute' in file.name else 'K3' if 'Fixup' in file.name else 'K4'
    raw=subprocess.check_output(['/usr/local/cuda-13.0/bin/cuobjdump','--dump-resource-usage',str(file)],text=True)
    function=next(line.split('Function ',1)[1].rstrip(':') for line in raw.splitlines() if 'Function ' in line)
    module,=check(cu.cuModuleLoad(str(file).encode()));fun,=check(cu.cuModuleGetFunction(module,function.encode()))
    block={'K1':128,'K2':384,'K3':256,'K4':384}[k]
    # K1 exact source layout; K3 minimum payload is sufficient to prove <=1.
    dyn={'K1':24960,'K2':0,'K3':163840,'K4':0}[k]
    if dyn>49152:check(cu.cuFuncSetAttribute(fun,cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,196608))
    count,=check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(fun,block,dyn))
    records.append({'kernel':k,'threads':block,'dynamic_smem_query':dyn,'max_active_CTA_per_SM':count,'note':'K1 exact source dynamic size; K2/K4 zero dynamic query is upper bound; K3 minimum payload query is upper bound; no kernel launched.'})
    check(cu.cuModuleUnload(module))
records.sort(key=lambda x:x['kernel']);Path(a.output).write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
check(cu.cuDevicePrimaryCtxRelease(device))
