from pathlib import Path
import hashlib,json,tarfile
O=Path(__file__).resolve().parent
groups={'R3_V2_SCREEN01_EVIDENCE':['r3_v2_screen01','r3_v2_screen01_telemetry','guard_r3_v2_screen01','R3_V2_SCREEN_NCU_ADMISSION.json','R3_V2_SCREEN_COMMAND.json'],
        'R3_V2_NCU01_EVIDENCE':['r3_v2_ncu_S_G128_LOAD_01','guard_r3_v2_ncu01','R3_V2_SCREEN_NCU_ADMISSION.json']}
for name,items in groups.items():
    out=O/(name+'.tgz')
    assert not out.exists()
    with tarfile.open(out,'w:gz') as t:
        for x in items:t.add(O/x,arcname=x)
    d={'file':out.name,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    (O/(name+'_MANIFEST.json')).write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
