from __future__ import annotations

import pytest
import torch

from op12_gdn import gdn_prefill_fp8
from op12_gdn.kernel_contract import (
    REGISTER_TOTAL,
    SMEM_RESERVED_BYTES,
    TMEM_TOTAL_COLUMNS,
    assert_resource_contract,
)
from conftest import fp8_randn


def _valid_tnd():
    q = fp8_randn((16, 2, 128))
    k = fp8_randn((16, 2, 128))
    v = fp8_randn((16, 2, 128))
    g = torch.full((16, 2), 0.99, dtype=torch.float32, device="cuda")
    beta = torch.full((16, 2), 0.2, dtype=torch.float32, device="cuda")
    cu = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    return q, k, v, g, beta, cu


def test_resource_contract():
    assert_resource_contract()
    assert REGISTER_TOTAL == 63_488
    assert SMEM_RESERVED_BYTES >= 35 * 1024
    assert TMEM_TOTAL_COLUMNS == 448


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda q, k, v, g, b, cu: (q.float(), k, v, g, b, cu), "E4M3"),
        (lambda q, k, v, g, b, cu: (q[..., :64], k[..., :64], v[..., :64], g, b, cu), "D must be 128"),
        (lambda q, k, v, g, b, cu: (q, k, v, g.half(), b, cu), "gate must be FP32"),
        (lambda q, k, v, g, b, cu: (q, k, v, g, b, cu.long()), "cu_seqlens must be int32"),
        (lambda q, k, v, g, b, cu: (q, k, v, g, b, None), "TND requires"),
    ],
)
def test_reject_invalid_contract(mutation, match):
    args = mutation(*_valid_tnd())
    with pytest.raises(ValueError, match=match):
        gdn_prefill_fp8(*args[:5], cu_seqlens=args[5])


def test_bnsd_rejects_cu_seqlens():
    q = fp8_randn((1, 1, 16, 128))
    g = torch.ones((1, 1, 16), dtype=torch.float32, device="cuda")
    cu = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="BNSD does not accept"):
        gdn_prefill_fp8(q, q, q, g, g, cu_seqlens=cu)


def test_reject_illegal_head_relation():
    q = fp8_randn((16, 4, 128))
    k = fp8_randn((16, 2, 128))
    v = fp8_randn((16, 4, 128))
    g = torch.ones((16, 4), dtype=torch.float32, device="cuda")
    cu = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="head relation"):
        gdn_prefill_fp8(q, k, v, g, g, cu_seqlens=cu)
