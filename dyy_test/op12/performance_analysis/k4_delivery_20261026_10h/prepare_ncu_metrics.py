"""Select only queried GB100 metrics; no collection or CUDA imports."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--query',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=list(csv.DictReader((a.query/'GB100_RELEVANT_SUFFIXES.txt').read_text().splitlines()))
available={r['Metric name']:r for r in rows}
bases=['l1tex__data_pipe_lsu_wavefronts','l1tex__data_pipe_lsu_wavefronts_cmd_read','l1tex__data_pipe_lsu_wavefronts_cmd_write',
 'l1tex__data_pipe_lsu_wavefronts_mem_lgds','l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_read','l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_write',
 'l1tex__data_pipe_lsu_wavefronts_mem_shared','l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_read','l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_write',
 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom',
 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_misc','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_umemsets','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ipa',
 'l1tex__data_pipe_lsu_wavefronts_mem_surface','l1tex__data_pipe_tc_wavefronts','l1tex__data_pipe_tc_wavefronts_mem_shared',
 'l1tex__data_pipe_tc_wavefronts_mem_shared_op_utccp','l1tex__data_pipe_tc_wavefronts_mem_shared_op_utcmma_matrix_a',
 'l1tex__data_pipe_tc_wavefronts_mem_shared_op_utcmma_matrix_b_scope_1cta','l1tex__data_pipe_tc_wavefronts_mem_shared_op_utcmma_matrix_b_scope_2cta',
 'l1tex__data_pipe_tex_wavefronts','l1tex__data_pipe_tex_wavefronts_mem_surface','l1tex__data_pipe_tex_wavefronts_mem_texture',
 'l1tex__data_bank_conflicts_pipe_lsu','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st',
 'l1tex__data_bank_conflicts_type_arbitration','l1tex__data_bank_reads','l1tex__data_bank_writes','l1tex__cycles_active','l1tex__cycles_elapsed',
 'l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld','l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st']
wanted=[b+'.sum' for b in bases]
for b in ['l1tex__data_pipe_lsu_wavefronts','l1tex__data_pipe_tc_wavefronts','l1tex__data_pipe_tex_wavefronts']:
    wanted += [b+'.avg.peak_sustained',b+'.avg.pct_of_peak_sustained_elapsed',b+'.avg.pct_of_peak_sustained_active']
wanted += ['l1tex__cycles_active.avg','l1tex__cycles_elapsed.avg']
# General traffic metrics are separately checked against the base query.
base_rows=list(csv.DictReader((a.query/'GB100_BASE_METRICS.txt').read_text().splitlines()))
allbase={r['Metric name'] for r in base_rows}
general=['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
 'lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum','dram__bytes_read.sum','dram__bytes_write.sum']
missing=[m for m in wanted if m not in available]
missing += [m for m in general if m.rsplit('.',1)[0] not in allbase]
report=dict(status='READY' if not missing else 'UNSUPPORTED_REQUESTED_METRIC',metrics=wanted+general,missing=missing,
 definitions={m:available[m] for m in wanted if m in available},
 caution='LSU is not the total Data parent including TC/TEX. Memory and command axes are alternate partitions. Hardware bank conflicts can include arbitration; do not add them to actual WF or independently sum replay costs with TC/TMA service.',model_enabled=False)
a.out.write_text(json.dumps(report,indent=2))
print(json.dumps(dict(status=report['status'],metric_count=len(report['metrics']),missing=missing),indent=2))
assert not missing
