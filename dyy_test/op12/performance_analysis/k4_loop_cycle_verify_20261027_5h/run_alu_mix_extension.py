from pathlib import Path
R=Path(__file__).resolve().parent
t=(R/'run_alu_mix_batch.py').read_text().replace("'alu_mix_'+a.phase","'alu_mix_v3_extension_'+a.phase").replace('run_alu_mix.py','run_alu_mix_v3.py').replace('1024,2048,4096,8192','2048,4096,8192,16384')
exec(compile(t,'run_alu_mix_extension_generated.py','exec'))
