from __future__ import annotations

import pytest
import torch

from op12_gdn import gdn_prefill_fp8
from conftest import fp8_randn, metrics


def _assert_accuracy(o, s, ref_o, ref_s):
    o_rel, o_cos = metrics(o, ref_o)
    s_rel, _ = metrics(s, ref_s)
    assert torch.isfinite(o).all() and torch.isfinite(s).all()
    assert o_rel <= 0.20
    assert o_cos >= 0.98
    assert s_rel <= 0.25


@pytest.mark.parametrize("seq_len", [1, 63, 64, 65, 127, 257])
def test_bnsd_lengths(seq_len):
    torch.manual_seed(seq_len)
    q, k, v = (fp8_randn((1, 1, seq_len, 128)) for _ in range(3))
    gate = torch.empty((1, 1, seq_len), dtype=torch.float32, device="cuda").uniform_(0.96, 1.0)
    beta = torch.empty_like(gate).uniform_(0.05, 0.3)
    (o, s), stats = gdn_prefill_fp8(q, k, v, gate, beta, output_final_state=True, use_cp=False, return_stats=True)
    ref_o, ref_s = gdn_prefill_fp8(q, k, v, gate, beta, output_final_state=True, use_cp=False, backend="reference")
    _assert_accuracy(o, s, ref_o, ref_s)
    assert set(stats["logical_gemm_counts"]) == {
        "1_Wkk_KKt", "2_Wqk_QKt", "3_KS_KSprev", "4_QS_QSprev",
        "5_NV_AinvV", "6_Ointra_WqkvNV", "7_dS_Ktdelta",
    }


def test_tnd_varlen_tail_predication():
    lengths = [1, 63, 64, 65, 127, 257]
    total = sum(lengths)
    torch.manual_seed(7)
    q, k, v = (fp8_randn((total, 1, 128)) for _ in range(3))
    gate = torch.empty((total, 1), dtype=torch.float32, device="cuda").uniform_(0.97, 1.0)
    beta = torch.empty_like(gate).uniform_(0.05, 0.25)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
    (o, s), stats = gdn_prefill_fp8(q, k, v, gate, beta, cu_seqlens=cu, output_final_state=True, use_cp=False, return_stats=True)
    ref_o, ref_s = gdn_prefill_fp8(q, k, v, gate, beta, cu_seqlens=cu, output_final_state=True, use_cp=False, backend="reference")
    _assert_accuracy(o, s, ref_o, ref_s)
    assert stats["padded_tail_rows"] > 0


@pytest.mark.parametrize("heads", [(8, 2, 2), (2, 2, 8)])
def test_gqa_gva(heads):
    hq, hk, hv = heads
    ho, seq_len = max(hq, hv), 64
    torch.manual_seed(sum(heads))
    q, k, v = fp8_randn((seq_len, hq, 128)), fp8_randn((seq_len, hk, 128)), fp8_randn((seq_len, hv, 128))
    gate = torch.empty((seq_len, ho), dtype=torch.float32, device="cuda").uniform_(0.98, 1.0)
    beta = torch.empty_like(gate).uniform_(0.05, 0.2)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    o, s = gdn_prefill_fp8(q, k, v, gate, beta, cu_seqlens=cu, output_final_state=True, use_cp=False)
    ref_o, ref_s = gdn_prefill_fp8(q, k, v, gate, beta, cu_seqlens=cu, output_final_state=True, use_cp=False, backend="reference")
    _assert_accuracy(o, s, ref_o, ref_s)


def test_initial_state_and_zero_inputs():
    seq_len = 65
    q = torch.zeros((seq_len, 1, 128), dtype=torch.float8_e4m3fn, device="cuda")
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    gate = torch.ones((seq_len, 1), dtype=torch.float32, device="cuda")
    beta = torch.ones_like(gate)
    initial = torch.randn((1, 1, 128, 128), dtype=torch.float16, device="cuda") * 0.01
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    o, s = gdn_prefill_fp8(q, k, v, gate, beta, initial_state=initial, cu_seqlens=cu, output_final_state=True, use_cp=False)
    assert torch.count_nonzero(o) == 0
    assert torch.equal(s, initial)


def test_cp_auto_semantic_route():
    seq_len = 4096
    q, k, v = (fp8_randn((seq_len, 1, 128), scale=0.01) for _ in range(3))
    gate = torch.full((seq_len, 1), 0.9999, dtype=torch.float32, device="cuda")
    beta = torch.full_like(gate, 0.05)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    (_, _), stats = gdn_prefill_fp8(q, k, v, gate, beta, cu_seqlens=cu, output_final_state=True, use_cp="auto", return_stats=True)
    assert stats["route"] == "cp_semantic_4096"
