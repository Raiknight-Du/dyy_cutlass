from pathlib import Path
R=Path(__file__).resolve().parent
t=(R/'run_alu_mix_batch.py').read_text().replace("'alu_mix_'+a.phase","'alu_mix_v2_'+a.phase").replace('run_alu_mix.py','run_alu_mix_v2.py')
exec(compile(t,'run_alu_mix_v2_batch_generated.py','exec'))
