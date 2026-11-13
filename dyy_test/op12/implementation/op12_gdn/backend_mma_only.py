"""FlashInfer GDN correctness model with only the seven core MMAs changed to FP8.

The public tensors and every non-MMA operation keep the archived FlashInfer
SM100 contract: packed TND FP16/BF16 Q/K/V and output, FP32 gate/beta and
prefix-gate math, chunk size 64, head size 128, and public state layout
``[N,H,V,K]``.  Immediately before each of the seven logical tensor-core
matrix multiplications, both operands are directly converted to E4M3 and the
mandatory ``torch._scaled_mm`` scales are literal one.  The FP8 temporaries do
not replace the public or inter-epilogue tensor dtypes.

This is an unfused numerical experiment.  It models the requested one-variable
change but is not presented as a compiled modification of FlashInfer's fused
CuTe kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .contracts import CHUNK_SIZE, HEAD_SIZE
from .quantization import QuantizationStats, fp8_unit_scale_mm, quantize_e4m3_no_scale


IO_DTYPES = (torch.float16, torch.bfloat16)
STATE_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


@dataclass
class MmaOnlyExecutionStats:
    logical_gemm_counts: dict[str, int]
    operand_quantization: QuantizationStats
    padded_tail_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": "mma_only_fp8_unfused",
            "changed_dimension": "seven_core_mma_operand_dtype_only",
            "mma_operand_dtype": "E4M3",
            "mma_operand_scales": "literal_one",
            "mma_accumulator_dtype": "FP32",
            "public_io_dtype": "unchanged_fp16_or_bf16",
            "non_mma_precision": "flashinfer_original_contract",
            "logical_gemm_counts": dict(self.logical_gemm_counts),
            "padded_tail_rows": self.padded_tail_rows,
            "mma_operand_quantization": self.operand_quantization.summarize(),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _head_index(output_head: int, input_heads: int, output_heads: int) -> int:
    return output_head // (output_heads // input_heads)


def _validate_heads(hq: int, hk: int, hv: int) -> None:
    mha = hq == hk == hv
    gqa = hq > hk and hk == hv and hq % hk == 0
    gva = hv > hq and hq == hk and hv % hq == 0
    _require(
        mha or gqa or gva,
        "head relation must be MHA, GQA (Hq > Hk = Hv), or GVA (Hv > Hq = Hk)",
    )


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor | None,
    beta: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require(q.is_cuda and k.is_cuda and v.is_cuda, "Q/K/V must be CUDA tensors")
    _require(q.device == k.device == v.device, "Q/K/V must be on the same CUDA device")
    _require(q.ndim == k.ndim == v.ndim == 3, "FlashInfer original input layout is packed TND rank 3")
    _require(q.dtype in IO_DTYPES, f"Q must be FP16 or BF16, got {q.dtype}")
    _require(k.dtype == q.dtype and v.dtype == q.dtype, "Q/K/V dtypes must match")
    _require(q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Q/K/V must be contiguous")
    _require(q.shape[0] == k.shape[0] == v.shape[0], "Q/K/V total-token dimensions must match")
    _require(q.shape[-1] == k.shape[-1] == v.shape[-1] == HEAD_SIZE, "head size must be 128")
    major, minor = torch.cuda.get_device_capability(q.device)
    _require((major, minor) == (10, 0), f"mma-only op12 requires B200 SM100, got {major}.{minor}")

    total, hq, _ = q.shape
    hk, hv = k.shape[1], v.shape[1]
    _validate_heads(hq, hk, hv)
    ho = max(hq, hv)
    _require(cu_seqlens is not None, "cu_seqlens is required for FlashInfer packed TND")
    assert cu_seqlens is not None
    _require(cu_seqlens.is_cuda and cu_seqlens.device == q.device, "cu_seqlens must be on the Q device")
    _require(cu_seqlens.dtype in (torch.int32, torch.int64), "cu_seqlens must be int32 or int64")
    _require(cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 2, "cu_seqlens must have shape [B+1]")
    cu_host = cu_seqlens.detach().cpu()
    _require(int(cu_host[0]) == 0 and int(cu_host[-1]) == total, "cu_seqlens endpoints must be 0 and T")
    _require(bool((cu_host[1:] >= cu_host[:-1]).all()), "cu_seqlens must be monotonic")
    batch = cu_seqlens.numel() - 1

    expected_scalar_shape = (total, ho)
    if gate is None:
        gate = torch.ones(expected_scalar_shape, dtype=torch.float32, device=q.device)
    if beta is None:
        beta = torch.ones(expected_scalar_shape, dtype=torch.float32, device=q.device)
    for name, x in (("gate", gate), ("beta", beta)):
        _require(x.is_cuda and x.device == q.device, f"{name} must be on the Q device")
        _require(x.dtype == torch.float32, f"{name} must be FP32")
        _require(tuple(x.shape) == expected_scalar_shape, f"{name} must have shape {expected_scalar_shape}")
        _require(x.is_contiguous(), f"{name} must be contiguous")
        _require(bool(torch.isfinite(x).all()), f"{name} contains NaN or Inf")
    _require(bool((gate >= 0).all()), "gate must be non-negative")

    if initial_state is not None:
        _require(initial_state.is_cuda and initial_state.device == q.device, "initial_state must be on the Q device")
        _require(initial_state.dtype in STATE_DTYPES, f"unsupported state dtype {initial_state.dtype}")
        _require(tuple(initial_state.shape) == (batch, ho, HEAD_SIZE, HEAD_SIZE), "initial_state must be [B,H,V,K]")
        _require(initial_state.stride(-1) == 1, "initial_state K dimension must be contiguous")

    return batch, total, hq, hk, hv, gate, beta, cu_seqlens.to(torch.int32)


def _pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    if x.shape[0] == rows:
        return x
    out = torch.zeros((rows, x.shape[1]), dtype=x.dtype, device=x.device)
    out[: x.shape[0]].copy_(x)
    return out


def _mma_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    stats: MmaOnlyExecutionStats,
    gemm_name: str,
    a_name: str,
    b_name: str,
) -> torch.Tensor:
    """Change only this matrix multiplication's operand dtype to E4M3."""

    aq = quantize_e4m3_no_scale(a, a_name, stats.operand_quantization).contiguous()
    bq = quantize_e4m3_no_scale(b, b_name, stats.operand_quantization).contiguous()
    stats.logical_gemm_counts[gemm_name] = stats.logical_gemm_counts.get(gemm_name, 0) + 1
    return fp8_unit_scale_mm(aq, bq)


@torch.inference_mode()
def chunk_gated_delta_rule_mma_only_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_cp: bool = False,
    return_stats: bool = False,
):
    """Run the FlashInfer non-CP chunked algorithm with only seven FP8 MMAs.

    This correctness backend intentionally accepts the original packed-TND
    FP16/BF16 public contract.  CP, checkpoints, and indexed state pools remain
    properties of the untouched upstream implementation and are outside this
    unfused single-variable experiment.
    """

    _require(use_cp is False, "mma-only correctness backend validates the original non-CP chunk path")
    batch, total, hq, hk, hv, gate, beta, cu = _validate_inputs(
        q, k, v, g, beta, cu_seqlens, initial_state
    )
    ho = max(hq, hv)
    score_scale = 1.0 / math.sqrt(HEAD_SIZE) if scale is None or scale == 0.0 else float(scale)
    _require(math.isfinite(score_scale), "scale must be finite")

    out = torch.empty((total, ho, HEAD_SIZE), dtype=q.dtype, device=q.device)
    # FlashInfer permits the persistent state dtype to be selected independently
    # from Q/K/V.  In this compact API we preserve an explicitly supplied
    # initial-state dtype; the no-initial-state default remains FP32.
    final_state_dtype = (
        torch.float32 if initial_state is None else initial_state.dtype
    )
    final_state = torch.empty(
        (batch, ho, HEAD_SIZE, HEAD_SIZE),
        dtype=final_state_dtype,
        device=q.device,
    )
    stats = MmaOnlyExecutionStats({}, QuantizationStats())
    cu_host = cu.cpu().tolist()

    for batch_idx in range(batch):
        seq_start, seq_end = int(cu_host[batch_idx]), int(cu_host[batch_idx + 1])
        seq_len = seq_end - seq_start
        for head in range(ho):
            q_head = _head_index(head, hq, ho)
            k_head = _head_index(head, hk, ho)
            v_head = _head_index(head, hv, ho)
            if initial_state is None:
                state_kv = torch.zeros((HEAD_SIZE, HEAD_SIZE), dtype=torch.float32, device=q.device)
            else:
                # Public FlashInfer state is [V,K]; mathematical state is [K,V].
                state_kv = initial_state[batch_idx, head].transpose(0, 1).float().clone()

            for local_start in range(0, seq_len, CHUNK_SIZE):
                valid = min(CHUNK_SIZE, seq_len - local_start)
                padded = ((valid + 15) // 16) * 16
                stats.padded_tail_rows += padded - valid
                token_start = seq_start + local_start
                token_end = token_start + valid
                q_tile = _pad_rows(q[token_start:token_end, q_head], padded)
                k_tile = _pad_rows(k[token_start:token_end, k_head], padded)
                v_tile = _pad_rows(v[token_start:token_end, v_head], padded)

                alpha = torch.ones((padded,), dtype=torch.float32, device=q.device)
                update = torch.zeros((padded,), dtype=torch.float32, device=q.device)
                alpha[:valid] = gate[token_start:token_end, head]
                update[:valid] = beta[token_start:token_end, head]
                cumulative_log = torch.cumsum(torch.log(alpha + 1.0e-10), dim=0)
                cumulative = torch.exp(cumulative_log)
                transfer = torch.exp(cumulative_log[:, None] - cumulative_log[None, :])
                lower = torch.tril(torch.ones((padded, padded), dtype=torch.bool, device=q.device))

                wkk = _mma_fp8(
                    k_tile,
                    k_tile.transpose(0, 1),
                    stats=stats,
                    gemm_name="1_Wkk_KKt",
                    a_name="K_for_Wkk",
                    b_name="Kt_for_Wkk",
                )
                strict_lower = torch.tril(torch.ones_like(wkk, dtype=torch.bool), diagonal=-1)
                a = torch.eye(padded, dtype=torch.float32, device=q.device)
                a.add_(torch.where(strict_lower, update[:, None] * transfer * wkk, torch.zeros_like(wkk)))
                # Keep the original FlashInfer reference precision outside MMA:
                # inverse/beta construction in FP32, then publish in io_dtype.
                a_inv_beta = (torch.linalg.inv(a) * update[None, :]).to(q.dtype)

                wqk = _mma_fp8(
                    q_tile,
                    k_tile.transpose(0, 1),
                    stats=stats,
                    gemm_name="2_Wqk_QKt",
                    a_name="Q_for_Wqk",
                    b_name="Kt_for_Wqk",
                )
                # The original SM100 epilogue publishes this operand in io_dtype.
                wqkv = torch.where(
                    lower, transfer * wqk * score_scale, torch.zeros_like(wqk)
                ).to(q.dtype)

                output_halves: list[torch.Tensor] = []
                next_state_halves: list[torch.Tensor] = []
                for value_start in (0, 64):
                    state_half = state_kv[:, value_start : value_start + 64]
                    state_operand = state_half.to(q.dtype)
                    ks = _mma_fp8(
                        k_tile,
                        state_operand,
                        stats=stats,
                        gemm_name="3_KS_KSprev",
                        a_name="K_for_KS",
                        b_name="state_for_KS",
                    )
                    qs = _mma_fp8(
                        q_tile,
                        state_operand,
                        stats=stats,
                        gemm_name="4_QS_QSprev",
                        a_name="Q_for_QS",
                        b_name="state_for_QS",
                    )

                    # Match the original SM100 residual boundary: KS is accumulated
                    # in FP32, gate-scaled, then converted to io_dtype before V-KS.
                    residual = (
                        v_tile[:, value_start : value_start + 64]
                        - (cumulative[:, None] * ks).to(q.dtype)
                    ).to(q.dtype)
                    nv = _mma_fp8(
                        a_inv_beta,
                        residual,
                        stats=stats,
                        gemm_name="5_NV_AinvResidual",
                        a_name="AinvBeta_for_NV",
                        b_name="residual_for_NV",
                    )
                    nv_operand = nv.to(q.dtype)
                    o_intra = _mma_fp8(
                        wqkv,
                        nv_operand,
                        stats=stats,
                        gemm_name="6_Ointra_WqkvNV",
                        a_name="Wqkv_for_O",
                        b_name="NV_for_O",
                    )
                    output_halves.append(
                        o_intra + cumulative[:, None] * qs * score_scale
                    )

                    decay_to_end = torch.exp(cumulative_log[valid - 1] - cumulative_log)
                    decay_to_end[valid:] = 0.0
                    delta_operand = (decay_to_end[:, None] * nv).to(q.dtype)
                    ds = _mma_fp8(
                        k_tile.transpose(0, 1),
                        delta_operand,
                        stats=stats,
                        gemm_name="7_dS_Ktdelta",
                        a_name="Kt_for_dS",
                        b_name="delta_for_dS",
                    )
                    next_state_halves.append(cumulative[valid - 1] * state_half + ds)

                out[token_start:token_end, head] = torch.cat(output_halves, dim=1)[:valid].to(q.dtype)
                state_kv = torch.cat(next_state_halves, dim=1)

            final_state[batch_idx, head] = state_kv.transpose(0, 1).to(
                final_state_dtype
            )

    primary = (out, final_state) if output_final_state else out
    return (primary, stats.as_dict()) if return_stats else primary
