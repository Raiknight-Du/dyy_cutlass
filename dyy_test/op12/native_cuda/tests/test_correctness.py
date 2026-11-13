from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from op12_native_gdn import chunk_gated_delta_rule as native_gdn

FLASHINFER_ROOT = Path(
    os.environ.get(
        "FLASHINFER_ROOT",
        "/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/upstreams/flashinfer",
    )
)
sys.path.insert(0, str(FLASHINFER_ROOT))
from flashinfer.gdn_prefill import chunk_gated_delta_rule as flashinfer_gdn


def make_case(seq_lens, heads, dtype, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    hq, hk, hv = heads
    total = sum(seq_lens)
    ho = max(hq, hv)
    q = torch.randn(total, hq, 128, device=device, dtype=dtype)
    k = torch.nn.functional.normalize(
        torch.randn(total, hk, 128, device=device, dtype=torch.float32), dim=-1
    ).to(dtype)
    v = torch.randn(total, hv, 128, device=device, dtype=dtype)
    g = torch.rand(total, ho, device=device, dtype=torch.float32)
    beta = torch.rand(total, ho, device=device, dtype=torch.float32)
    cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), device=device)
    state = torch.randn(len(seq_lens), ho, 128, 128, device=device) * 0.01
    return q, k, v, g, beta, cu, state


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [(2, 2, 2), (8, 2, 2), (2, 2, 8)])
@pytest.mark.parametrize("seq_lens", [[1], [63, 65], [64, 127]])
def test_against_flashinfer(dtype, heads, seq_lens):
    q, k, v, g, beta, cu, state = make_case(seq_lens, heads, dtype)
    kwargs = dict(
        g=g,
        beta=beta,
        scale=1.0 / math.sqrt(128),
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=False,
        use_cp=False,
    )
    out_native, state_native = native_gdn(q, k, v, **kwargs)
    out_upstream, state_upstream = flashinfer_gdn(q, k, v, **kwargs)
    if dtype == torch.float16:
        torch.testing.assert_close(out_native, out_upstream, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(state_native, state_upstream, atol=2e-3, rtol=2e-3)
    else:
        torch.testing.assert_close(out_native, out_upstream, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(state_native, state_upstream, atol=1e-2, rtol=5e-3)


def test_state_pool_and_checkpoints():
    q, k, v, g, beta, cu, packed_state = make_case([64, 128], (2, 2, 2), torch.float16)
    pool = torch.zeros(5, 2, 128, 128, device="cuda")
    indices = torch.tensor([1, 4], device="cuda", dtype=torch.int32)
    pool[indices.long()] = packed_state
    checkpoints = torch.empty(3, 2, 128, 128, device="cuda")
    checkpoint_starts = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int64)
    out, final_pool = native_gdn(
        q,
        k,
        v,
        g,
        beta,
        initial_state=pool,
        output_final_state=True,
        output_state=pool,
        cu_seqlens=cu,
        state_indices=indices,
        state_checkpoints=checkpoints,
        checkpoint_cu_starts=checkpoint_starts,
        checkpoint_every_n_tokens=64,
        use_cp=False,
    )
    assert torch.isfinite(out).all()
    assert torch.isfinite(final_pool[indices.long()]).all()
    assert torch.isfinite(checkpoints).all()


@pytest.mark.parametrize(
    "state_dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ],
)
def test_state_dtype_against_flashinfer(state_dtype):
    q, k, v, g, beta, cu, state_f32 = make_case([64], (2, 2, 2), torch.float16)
    state = state_f32.to(state_dtype)
    native_out_state = torch.empty_like(state)
    upstream_out_state = torch.empty_like(state)
    common = dict(
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
        use_cp=False,
    )
    out_native, state_native = native_gdn(
        q, k, v, output_state=native_out_state, **common
    )
    out_upstream, state_upstream = flashinfer_gdn(
        q, k, v, output_state=upstream_out_state, **common
    )
    output_atol = 4e-3 if state_dtype != torch.float8_e5m2 else 2e-2
    state_atol = 3e-3 if state_dtype in (torch.float32, torch.float16) else 2e-2
    if state_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        state_atol = 1.0
    torch.testing.assert_close(out_native, out_upstream, atol=output_atol, rtol=3e-3)
    torch.testing.assert_close(
        state_native.float(), state_upstream.float(), atol=state_atol, rtol=1e-2
    )


def test_state_pool_against_flashinfer():
    q, k, v, g, beta, cu, packed = make_case([31, 65], (2, 2, 2), torch.float16)
    indices = torch.tensor([1, 4], device="cuda", dtype=torch.int32)
    native_pool = torch.zeros(5, 2, 128, 128, device="cuda")
    native_pool[indices.long()] = packed
    upstream_pool = native_pool.clone()
    out_native, _ = native_gdn(
        q,
        k,
        v,
        g,
        beta,
        initial_state=native_pool,
        output_final_state=True,
        output_state=native_pool,
        cu_seqlens=cu,
        state_indices=indices,
        use_cp=False,
    )
    out_upstream, _ = flashinfer_gdn(
        q,
        k,
        v,
        g,
        beta,
        initial_state=upstream_pool,
        output_final_state=True,
        output_state=upstream_pool,
        cu_seqlens=cu,
        state_indices=indices,
        use_cp=False,
    )
    torch.testing.assert_close(out_native, out_upstream, atol=3e-3, rtol=2e-3)
    torch.testing.assert_close(native_pool, upstream_pool, atol=2e-3, rtol=2e-3)


def test_checkpoints_against_flashinfer():
    q, k, v, g, beta, cu, state = make_case([64, 128], (2, 2, 2), torch.float16)
    native_checkpoints = torch.empty(3, 2, 128, 128, device="cuda")
    upstream_checkpoints = torch.empty_like(native_checkpoints)
    checkpoint_starts = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int64)
    common = dict(
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
        checkpoint_cu_starts=checkpoint_starts,
        checkpoint_every_n_tokens=64,
        use_cp=False,
    )
    out_native, state_native = native_gdn(
        q, k, v, state_checkpoints=native_checkpoints, **common
    )
    out_upstream, state_upstream = flashinfer_gdn(
        q, k, v, state_checkpoints=upstream_checkpoints, **common
    )
    torch.testing.assert_close(out_native, out_upstream, atol=3e-3, rtol=2e-3)
    torch.testing.assert_close(state_native, state_upstream, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        native_checkpoints, upstream_checkpoints, atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [(2, 2, 2), (8, 2, 2), (2, 2, 8)])
@pytest.mark.parametrize("seq_lens", [[256], [511, 513]])
def test_cp_semantics_against_flashinfer(dtype, heads, seq_lens):
    q, k, v, g, beta, cu, state = make_case(seq_lens, heads, dtype)
    common = dict(
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
        use_cp=True,
    )
    out_native, state_native = native_gdn(q, k, v, **common)
    out_upstream, state_upstream = flashinfer_gdn(q, k, v, **common)
    if dtype == torch.float16:
        torch.testing.assert_close(out_native, out_upstream, atol=5e-3, rtol=4e-3)
        torch.testing.assert_close(state_native, state_upstream, atol=4e-3, rtol=4e-3)
    else:
        torch.testing.assert_close(out_native, out_upstream, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(state_native, state_upstream, atol=2e-2, rtol=2e-2)


def test_zero_length_sequences_against_flashinfer():
    q, k, v, g, beta, _, state = make_case([64], (2, 2, 2), torch.float16)
    cu = torch.tensor([0, 0, 64, 64], device="cuda", dtype=torch.int32)
    state3 = torch.cat((state, state, state), dim=0)
    common = dict(
        g=g,
        beta=beta,
        initial_state=state3,
        output_final_state=True,
        cu_seqlens=cu,
        use_cp=False,
    )
    out_native, state_native = native_gdn(q, k, v, **common)
    out_upstream, state_upstream = flashinfer_gdn(q, k, v, **common)
    torch.testing.assert_close(out_native, out_upstream, atol=3e-3, rtol=2e-3)
    torch.testing.assert_close(state_native, state_upstream, atol=2e-3, rtol=2e-3)
