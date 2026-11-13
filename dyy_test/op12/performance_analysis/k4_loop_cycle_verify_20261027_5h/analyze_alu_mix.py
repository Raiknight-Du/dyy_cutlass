from pathlib import Path
import sys
R=Path(__file__).resolve().parent
t=(R/'analyze_formal.py').read_text()
t=t.replace("else 'formal_v2'","else 'alu_mix_v2_formal'")
t=t.replace("root.glob('kind*_t*/EVENT_RAW.csv')","root.glob('t*/RAW.csv')")
t=t.replace('cta_cycle_median','clock_cycles')
t=t.replace('Matched native family loop + control + final completion, not bare primitive rate or K4 phase','Register-only PACK + three-live-source IADD3 same-CUBIN mix; mechanism and conditional service validation, not K4 phase weight')
exec(compile(t,'analyze_alu_mix_generated.py','exec'))
