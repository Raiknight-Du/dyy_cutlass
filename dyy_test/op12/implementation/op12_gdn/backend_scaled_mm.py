"""Executable seven-GEMM FP8 backend for correctness and B200 bring-up.

This backend maps every logical GEMM to ``torch._scaled_mm`` with E4M3
operands and FP32 output.  On B200 this dispatches the CUDA 13 SM100 FP8
Tensor-Core path.  It is intentionally unfused; the CuTe fused kernel is a
separate optimization stage and is not silently represented by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .contracts import CHUNK_SIZE, FP8_DTYPE, Problem
from .quantization import (
    QuantizationStats,
    fp8_scaled_mm,
    fp8_unit_scale_mm,
    quantize_e4m3_no_scale,
    quantize_e4m3_tile,
)


QuantizationMode = Literal["tile_scale", "unit_scale"]


@dataclass
class ExecutionStats:
    route: str
    logical_gemm_counts: dict[str, int]
    quantization: QuantizationStats
    quantization_mode: QuantizationMode = "tile_scale"
    padded_tail_rows: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": (
                "unit_scale_fp8_unfused"
                if self.quantization_mode == "unit_scale"
                else "scaled_mm_unfused"
            ),
            "quantization_mode": self.quantization_mode,
            "mma_operand_scales": "literal_one"
            if self.quantization_mode == "unit_scale"
            else "tile_amax_over_448",
            "route": self.route,
            "logical_gemm_counts": dict(self.logical_gemm_counts),
            "padded_tail_rows": self.padded_tail_rows,
            "quantization": self.quantization.summarize(),
        }


def _head_index(output_head: int, input_heads: int, output_heads: int) -> int:
    return output_head // (output_heads // input_heads)


def _inc(stats: ExecutionStats, name: str) -> None:
    stats.logical_gemm_counts[name] = stats.logical_gemm_counts.get(name, 0) + 1


def _raw_mm(a: torch.Tensor, b: torch.Tensor, stats: ExecutionStats, name: str) -> torch.Tensor:
    _inc(stats, name)
    return fp8_unit_scale_mm(a, b)


def _quantize_tile(
    x: torch.Tensor,
    name: str,
    stats: ExecutionStats,
) -> tuple[torch.Tensor, torch.Tensor | float]:
    if stats.quantization_mode == "unit_scale":
        return quantize_e4m3_no_scale(x, name, stats.quantization), 1.0
    return quantize_e4m3_tile(x, name, stats.quantization)


def _scaled_tile_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_name: str,
    b_name: str,
    stats: ExecutionStats,
    gemm_name: str,
    *,
    a_is_raw_fp8: bool = False,
    b_is_raw_fp8: bool = False,
) -> torch.Tensor:
    if a_is_raw_fp8:
        aq, sa = a, 1.0
    else:
        aq, sa = _quantize_tile(a, a_name, stats)
    if b_is_raw_fp8:
        bq, sb = b, 1.0
    else:
        bq, sb = _quantize_tile(b, b_name, stats)
    _inc(stats, gemm_name)
    if stats.quantization_mode == "unit_scale":
        return fp8_unit_scale_mm(aq, bq)
    return fp8_scaled_mm(aq, bq, sa, sb)


def _load_chunk(
    x: torch.Tensor,
    p: Problem,
    b: int,
    head: int,
    seq_start: int,
    local_start: int,
    valid: int,
) -> torch.Tensor:
    if p.layout == "BNSD":
        return x[b, head, local_start : local_start + valid, :]
    packed_start = seq_start + local_start
    return x[packed_start : packed_start + valid, head, :]


def _load_scalar_chunk(
    x: torch.Tensor,
    p: Problem,
    b: int,
    head: int,
    seq_start: int,
    local_start: int,
    valid: int,
) -> torch.Tensor:
    if p.layout == "BNSD":
        return x[b, head, local_start : local_start + valid]
    packed_start = seq_start + local_start
    return x[packed_start : packed_start + valid, head]


def _pad_rows_fp8(x: torch.Tensor, rows: int) -> torch.Tensor:
    if x.shape[0] == rows:
        return x
    out = torch.zeros((rows, x.shape[1]), dtype=FP8_DTYPE, device=x.device)
    out[: x.shape[0]].copy_(x)
    return out


@torch.inference_mode()
def run_scaled_mm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    p: Problem,
    *,
    initial_state: torch.Tensor | None,
    score_scale: float,
    use_cp: str | bool,
    quantization_mode: QuantizationMode = "tile_scale",
) -> tuple[torch.Tensor, torch.Tensor, ExecutionStats]:
    """Run chunked GDN with seven E4M3 GEMM groups and FP32 epilogues."""

    max_seq = int((p.cu_seqlens[1:] - p.cu_seqlens[:-1]).max().item()) if p.batch else 0
    auto_cp = p.batch * p.ho < 74 and max_seq >= 4096
    cp_requested = use_cp is True or (use_cp == "auto" and auto_cp)
    # This baseline preserves CP macroblock boundaries for validation, but it
    # is not the future multi-kernel parallel-prefix implementation.
    route = "cp_semantic_4096" if cp_requested else "chunked"
    stats = ExecutionStats(
        route=route,
        logical_gemm_counts={},
        quantization=QuantizationStats(),
        quantization_mode=quantization_mode,
    )
    if p.layout == "BNSD":
        out = torch.empty((p.batch, p.ho, p.sequence_length, 128), dtype=torch.float16, device=q.device)
    else:
        out = torch.empty((p.total_tokens, p.ho, 128), dtype=torch.float16, device=q.device)
    final_state = torch.empty((p.batch, p.ho, 128, 128), dtype=torch.float16, device=q.device)
    cu = p.cu_seqlens.detach().cpu().tolist()

    for b in range(p.batch):
        seq_start, seq_end = int(cu[b]), int(cu[b + 1])
        seq_len = seq_end - seq_start
        macro_starts = range(0, seq_len, 4096 if cp_requested else max(seq_len, 1))
        for h in range(p.ho):
            qh = _head_index(h, p.hq, p.ho)
            kh = _head_index(h, p.hk, p.ho)
            vh = _head_index(h, p.hv, p.ho)
            if initial_state is None:
                state_kv = torch.zeros((128, 128), dtype=torch.float32, device=q.device)
            else:
                state_kv = initial_state[b, h].transpose(0, 1).float().clone()

            for macro_start in macro_starts:
                macro_end = min(seq_len, macro_start + (4096 if cp_requested else seq_len))
                for local_start in range(macro_start, macro_end, CHUNK_SIZE):
                    valid = min(CHUNK_SIZE, macro_end - local_start)
                    # FP8 cuBLASLt requires M/N multiples of 16. Invalid rows
                    # have K=Q=V=beta=0 and gate=1, making their state delta
                    # exactly zero; only the valid prefix is stored.
                    padded = ((valid + 15) // 16) * 16
                    stats.padded_tail_rows += padded - valid
                    q_tile = _pad_rows_fp8(_load_chunk(q, p, b, qh, seq_start, local_start, valid), padded)
                    k_tile = _pad_rows_fp8(_load_chunk(k, p, b, kh, seq_start, local_start, valid), padded)
                    v_tile = _pad_rows_fp8(_load_chunk(v, p, b, vh, seq_start, local_start, valid), padded)
                    alpha = torch.ones((padded,), dtype=torch.float32, device=q.device)
                    update = torch.zeros((padded,), dtype=torch.float32, device=q.device)
                    alpha[:valid] = _load_scalar_chunk(gate, p, b, h, seq_start, local_start, valid)
                    update[:valid] = _load_scalar_chunk(beta, p, b, h, seq_start, local_start, valid)

                    log_alpha = torch.log(alpha + 1.0e-10)
                    cumulative_log = torch.cumsum(log_alpha, dim=0)
                    cumulative = torch.exp(cumulative_log)
                    transfer = torch.exp(cumulative_log[:, None] - cumulative_log[None, :])
                    lower = torch.tril(torch.ones((padded, padded), dtype=torch.bool, device=q.device))

                    # 1: K K^T (raw external E4M3 x E4M3).
                    wkk = _raw_mm(k_tile, k_tile.transpose(0, 1), stats, "1_Wkk_KKt")
                    strict_lower = torch.tril(torch.ones_like(wkk, dtype=torch.bool), diagonal=-1)
                    a = torch.eye(padded, dtype=torch.float32, device=q.device)
                    a.add_(torch.where(strict_lower, update[:, None] * transfer * wkk, torch.zeros_like(wkk)))
                    a_inv_beta = torch.linalg.inv(a) * update[None, :]

                    # 2: Q K^T (raw external E4M3 x E4M3).
                    wqk = _raw_mm(q_tile, k_tile.transpose(0, 1), stats, "2_Wqk_QKt")
                    wqkv = torch.where(lower, transfer * wqk * score_scale, torch.zeros_like(wqk))

                    # State is logically split into two 128x64 V half-tiles,
                    # matching the 2-SM DV ownership plan. The executable
                    # baseline runs the halves serially but keeps independent
                    # scales and write ranges.
                    o_parts: list[torch.Tensor] = []
                    state_parts: list[torch.Tensor] = []
                    for dv0 in (0, 64):
                        state_half = state_kv[:, dv0 : dv0 + 64]
                        state_q, state_s = _quantize_tile(state_half, "state", stats)

                        # 3/4: external raw K/Q x online-quantized state.
                        _inc(stats, "3_KS_KSprev")
                        ks = (
                            fp8_unit_scale_mm(k_tile, state_q)
                            if quantization_mode == "unit_scale"
                            else fp8_scaled_mm(k_tile, state_q, 1.0, state_s)
                        )
                        _inc(stats, "4_QS_QSprev")
                        qs = (
                            fp8_unit_scale_mm(q_tile, state_q)
                            if quantization_mode == "unit_scale"
                            else fp8_scaled_mm(q_tile, state_q, 1.0, state_s)
                        )

                        residual = v_tile[:, dv0 : dv0 + 64].float() - cumulative[:, None] * ks
                        # 5: online A^-1*beta x online residual.
                        nv = _scaled_tile_mm(
                            a_inv_beta,
                            residual,
                            "a_inverse",
                            "delta",
                            stats,
                            "5_NV_AinvV",
                        )

                        # 6: online Wqkv x online NV.
                        o_intra = _scaled_tile_mm(
                            wqkv,
                            nv,
                            "wqkv",
                            "nv",
                            stats,
                            "6_Ointra_WqkvNV",
                        )
                        o_parts.append(o_intra + cumulative[:, None] * qs * score_scale)

                        decay_to_end = torch.exp(cumulative_log[valid - 1] - cumulative_log)
                        decay_to_end[valid:] = 0.0
                        delta_state = decay_to_end[:, None] * nv
                        delta_q, delta_s = _quantize_tile(delta_state, "delta_state", stats)
                        # 7: raw K^T x online state residual.
                        _inc(stats, "7_dS_Ktdelta")
                        ds = (
                            fp8_unit_scale_mm(k_tile.transpose(0, 1), delta_q)
                            if quantization_mode == "unit_scale"
                            else fp8_scaled_mm(k_tile.transpose(0, 1), delta_q, 1.0, delta_s)
                        )
                        state_parts.append(cumulative[valid - 1] * state_half + ds)

                    o_chunk = torch.cat(o_parts, dim=1)[:valid].to(torch.float16)
                    state_kv = torch.cat(state_parts, dim=1)
                    if p.layout == "BNSD":
                        out[b, h, local_start : local_start + valid] = o_chunk
                    else:
                        packed_start = seq_start + local_start
                        out[packed_start : packed_start + valid, h] = o_chunk

            final_state[b, h] = state_kv.transpose(0, 1).to(torch.float16)
    return out, final_state, stats
