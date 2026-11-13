from __future__ import annotations

import pytest
import torch

from op12_gdn import chunk_gated_delta_rule_mma_only_fp8


@pytest.mark.parametrize("io_dtype", [torch.float16, torch.bfloat16])
def test_mma_only_keeps_public_dtype_and_reports_seven_groups(io_dtype):
    torch.manual_seed(1212)
    q = (torch.randn((65, 1, 128), device="cuda") * 0.1).to(io_dtype)
    k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).to(io_dtype)
    v = (torch.randn((65, 1, 128), device="cuda") * 0.1).to(io_dtype)
    gate = torch.empty((65, 1), dtype=torch.float32, device="cuda").uniform_(0.96, 1.0)
    beta = torch.empty_like(gate).uniform_(0.05, 0.3)
    cu = torch.tensor([0, 65], dtype=torch.int32, device="cuda")

    (out, state), stats = chunk_gated_delta_rule_mma_only_fp8(
        q,
        k,
        v,
        gate,
        beta,
        cu_seqlens=cu,
        output_final_state=True,
        return_stats=True,
    )

    assert out.dtype == io_dtype
    assert state.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()
    assert stats["backend"] == "mma_only_fp8_unfused"
    assert stats["mma_operand_scales"] == "literal_one"
    assert stats["public_io_dtype"] == "unchanged_fp16_or_bf16"
    assert set(stats["logical_gemm_counts"]) == {
        "1_Wkk_KKt",
        "2_Wqk_QKt",
        "3_KS_KSprev",
        "4_QS_QSprev",
        "5_NV_AinvResidual",
        "6_Ointra_WqkvNV",
        "7_dS_Ktdelta",
    }


def test_mma_only_rejects_fp8_public_input_and_bnsd():
    q = torch.zeros((16, 1, 128), dtype=torch.float8_e4m3fn, device="cuda")
    gate = torch.ones((16, 1), dtype=torch.float32, device="cuda")
    cu = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="FP16 or BF16"):
        chunk_gated_delta_rule_mma_only_fp8(q, q, q, gate, gate, cu_seqlens=cu)

    q4 = torch.zeros((1, 1, 16, 128), dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="packed TND"):
        chunk_gated_delta_rule_mma_only_fp8(q4, q4, q4, gate, gate, cu_seqlens=cu)
