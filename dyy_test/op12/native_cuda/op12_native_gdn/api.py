from __future__ import annotations

import math
import os
from typing import Literal, Optional, Tuple, Union

import torch

from .loader import load_extension
from .tcgen05_backend import get_tcgen05_entrypoint

_STATE_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


def _empty(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.empty((0,), device=device, dtype=dtype)


def _check_tensor(name: str, x: torch.Tensor, ndim: int) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {x.ndim}D")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@torch.inference_mode()
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    use_cp: Literal["auto"] | bool = "auto",
    state_indices: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """FlashInfer-compatible SM100 GDN prefill contract.

    ``use_cp=False`` selects the persistent recurrent kernel.  ``True`` selects
    the three-kernel affine-partition path (local M/N, prefix fixup, independent
    final prefill); ``"auto"`` follows the fixed upstream SM100 dispatch
    threshold.  The fixed SM100 upstream path does not pass
    ``use_qk_l2norm_in_kernel`` into its prefill kernel, so this implementation
    intentionally preserves that effective behavior.
    """
    if use_cp not in ("auto", True, False):
        raise ValueError(f'use_cp must be "auto", True, or False, got {use_cp!r}')
    if cu_seqlens is None:
        raise AssertionError("cu_seqlens is required for varlen mode")
    for name, x in (("q", q), ("k", k), ("v", v)):
        _check_tensor(name, x, 3)
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"q/k/v must be FP16 or BF16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k and v must have the same dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must be on the same device")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k and v must have the same total token count")
    if q.shape[2] != 128 or k.shape[2] != 128 or v.shape[2] != 128:
        raise ValueError("SM100 native GDN requires head_size=128")

    total_tokens, hq, _ = q.shape
    hk, hv = k.shape[1], v.shape[1]
    ho = max(hq, hv)
    if ho % hq or ho % hk or ho % hv:
        raise ValueError(f"illegal head ratio Hq/Hk/Hv={hq}/{hk}/{hv}")
    if not ((hq >= hv and hk == hv) or (hv >= hq and hk == hq)):
        raise ValueError(f"expected MHA/GQA/GVA head relation, got {hq}/{hk}/{hv}")

    _check_tensor("cu_seqlens", cu_seqlens, 1)
    if cu_seqlens.device != q.device:
        raise ValueError("cu_seqlens must be on the q device")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least [0, total_tokens]")
    cu32 = cu_seqlens.to(torch.int32)
    num_seqs = cu32.numel() - 1
    # Avoid a device-to-host validation sync in the hot path, matching upstream.

    if g is None:
        g = torch.ones((total_tokens, ho), device=q.device, dtype=torch.float32)
    if beta is None:
        beta = torch.ones((total_tokens, ho), device=q.device, dtype=torch.float32)
    for name, x in (("g", g), ("beta", beta)):
        _check_tensor(name, x, 2)
        if x.dtype != torch.float32 or tuple(x.shape) != (total_tokens, ho):
            raise ValueError(f"{name} must be FP32 [{total_tokens}, {ho}]")

    if output is None:
        output = torch.empty((total_tokens, ho, 128), device=q.device, dtype=q.dtype)
    _check_tensor("output", output, 3)
    if output.dtype != q.dtype or tuple(output.shape) != (total_tokens, ho, 128):
        raise ValueError("output shape/dtype mismatch")

    pool_rows = num_seqs
    if state_indices is not None:
        _check_tensor("state_indices", state_indices, 1)
        if state_indices.dtype != torch.int32 or state_indices.numel() != num_seqs:
            raise ValueError("state_indices must be int32 [num_seqs]")
        if initial_state is None:
            raise ValueError("state_indices requires an explicit initial_state pool")
        pool_rows = initial_state.shape[0]
    else:
        state_indices = _empty(q.device, torch.int32)

    if initial_state is not None:
        _check_tensor("initial_state", initial_state, 4)
        if initial_state.dtype not in _STATE_DTYPES:
            raise ValueError(f"unsupported initial_state dtype {initial_state.dtype}")
        if tuple(initial_state.shape[1:]) != (ho, 128, 128):
            raise ValueError("initial_state must be [N, Ho, 128, 128]")
        if state_indices.numel() == 0 and initial_state.shape[0] != num_seqs:
            raise ValueError("packed initial_state first dimension must equal num_seqs")
        initial_arg = initial_state
    else:
        initial_arg = _empty(q.device)

    if output_final_state:
        if output_state is None:
            if state_indices.numel() != 0:
                raise ValueError("state_indices requires an explicit output_state pool")
            output_state = torch.empty(
                (num_seqs, ho, 128, 128), device=q.device, dtype=torch.float32
            )
        _check_tensor("output_state", output_state, 4)
        if output_state.dtype not in _STATE_DTYPES:
            raise ValueError(f"unsupported output_state dtype {output_state.dtype}")
        if tuple(output_state.shape[1:]) != (ho, 128, 128):
            raise ValueError("output_state must be [N, Ho, 128, 128]")
        if state_indices.numel() == 0 and output_state.shape[0] != num_seqs:
            raise ValueError("packed output_state first dimension must equal num_seqs")
        output_state_arg = output_state
    else:
        output_state_arg = _empty(q.device)

    if checkpoint_every_n_tokens < 0 or checkpoint_every_n_tokens % 64 != 0:
        raise ValueError("checkpoint_every_n_tokens must be zero or a multiple of 64")
    if checkpoint_every_n_tokens:
        if state_checkpoints is None or checkpoint_cu_starts is None:
            raise ValueError("checkpoint tensors are required when checkpointing is enabled")
        _check_tensor("state_checkpoints", state_checkpoints, 4)
        _check_tensor("checkpoint_cu_starts", checkpoint_cu_starts, 1)
        if state_checkpoints.dtype not in _STATE_DTYPES:
            raise ValueError("unsupported state_checkpoints dtype")
        if tuple(state_checkpoints.shape[1:]) != (ho, 128, 128):
            raise ValueError("state_checkpoints must be [Nckpt, Ho, 128, 128]")
        if checkpoint_cu_starts.dtype != torch.int64:
            raise ValueError("checkpoint_cu_starts must be int64")
        checkpoints_arg = state_checkpoints
        checkpoint_starts_arg = checkpoint_cu_starts.to(torch.int32)
    else:
        if state_checkpoints is not None or checkpoint_cu_starts is not None:
            raise ValueError("checkpoint tensors must be None when checkpointing is disabled")
        checkpoints_arg = _empty(q.device)
        checkpoint_starts_arg = _empty(q.device, torch.int32)

    effective_scale = float(scale) if scale is not None and scale != 0 else 1.0 / math.sqrt(128)
    backend = os.environ.get("OP12_NATIVE_BACKEND", "tcgen05").lower()
    if backend not in ("tcgen05", "wmma", "scalar"):
        raise ValueError("OP12_NATIVE_BACKEND must be 'tcgen05', 'wmma', or 'scalar'")

    if backend == "tcgen05":
        upstream = get_tcgen05_entrypoint()
        return upstream(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=effective_scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output=output,
            output_state=output_state,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            use_cp=use_cp,
            state_indices=None if state_indices.numel() == 0 else state_indices,
        )

    ext = load_extension()
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    cp_heuristic = num_seqs * ho * 4 < sm_count
    will_use_cp = use_cp is True or (use_cp == "auto" and cp_heuristic)
    cp_rejection = checkpoint_every_n_tokens > 0 or state_indices.numel() != 0
    if will_use_cp and cp_rejection:
        if use_cp is True:
            raise ValueError("native CP currently excludes checkpoints and state_indices")
        will_use_cp = False

    if will_use_cp:
        parallel_heads = max(1, num_seqs * ho)
        target = max(1, sm_count // parallel_heads)
        target_parts = 1 << (target.bit_length() - 1)
        target_parts = min(target_parts, 16)
        average_tokens = max(1, math.ceil(total_tokens / num_seqs))
        cp_chunk_len = max(
            512,
            math.ceil(average_tokens / target_parts / 512) * 512,
        )
        max_partitions = max(1, math.ceil(total_tokens / cp_chunk_len))
        workspace_shape = (num_seqs, max_partitions, ho, 128, 128)
        workspace_m = torch.empty(workspace_shape, device=q.device, dtype=torch.float32)
        workspace_n = torch.empty_like(workspace_m)
        workspace_boundaries = torch.empty_like(workspace_m)
        ext.gdn_prefill_cp_cuda(
            q,
            k,
            v,
            g,
            beta,
            output,
            cu32,
            initial_arg,
            output_state_arg,
            workspace_m,
            workspace_n,
            workspace_boundaries,
            effective_scale,
            cp_chunk_len,
            max_partitions,
        )
    else:
        launch = ext.gdn_prefill_wmma_cuda if backend == "wmma" else ext.gdn_prefill_cuda
        launch(
            q,
            k,
            v,
            g,
            beta,
            output,
            cu32,
            initial_arg,
            output_state_arg,
            checkpoints_arg,
            checkpoint_starts_arg,
            state_indices,
            effective_scale,
            int(checkpoint_every_n_tokens),
            *(() if backend == "wmma" else (False,)),
        )
    if output_final_state:
        return output, output_state
    return output
