from __future__ import annotations

import torch

from op12_gdn import gdn_prefill_fp8


def test_unit_scale_backend_reports_literal_one_scales():
    torch.manual_seed(1201)
    q = (torch.randn((64, 1, 128), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn((64, 1, 128), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    gate = torch.empty((64, 1), dtype=torch.float32, device="cuda").uniform_(0.96, 1.0)
    beta = torch.empty_like(gate).uniform_(0.05, 0.3)
    cu = torch.tensor([0, 64], dtype=torch.int32, device="cuda")

    (out, state), stats = gdn_prefill_fp8(
        q,
        k,
        v,
        gate,
        beta,
        cu_seqlens=cu,
        output_final_state=True,
        backend="unit_scale_fp8",
        use_cp=False,
        return_stats=True,
    )

    assert torch.isfinite(out).all()
    assert torch.isfinite(state).all()
    assert stats["backend"] == "unit_scale_fp8_unfused"
    assert stats["quantization_mode"] == "unit_scale"
    assert stats["mma_operand_scales"] == "literal_one"
    assert set(stats["logical_gemm_counts"]) == {
        "1_Wkk_KKt",
        "2_Wqk_QKt",
        "3_KS_KSprev",
        "4_QS_QSprev",
        "5_NV_AinvV",
        "6_Ointra_WqkvNV",
        "7_dS_Ktdelta",
    }

