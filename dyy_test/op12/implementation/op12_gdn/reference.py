"""FP32 recurrent gold model for raw E4M3 op12 inputs."""

from __future__ import annotations

import math

import torch

from .contracts import Problem, validate_problem


def _head_index(output_head: int, input_heads: int, output_heads: int) -> int:
    return output_head // (output_heads // input_heads)


@torch.inference_mode()
def gdn_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-recurrent FP32 gold model, returning FP16 O and [B,H,V,K] state."""

    p = validate_problem(q, k, v, gate, beta, cu_seqlens, initial_state)
    score_scale = 1.0 / math.sqrt(128) if scale is None else float(scale)
    if p.layout == "BNSD":
        out = torch.empty((p.batch, p.ho, p.sequence_length, 128), dtype=torch.float16, device=q.device)
    else:
        out = torch.empty((p.total_tokens, p.ho, 128), dtype=torch.float16, device=q.device)
    final_state = torch.empty((p.batch, p.ho, 128, 128), dtype=torch.float16, device=q.device)
    cu = p.cu_seqlens.detach().cpu().tolist()

    for b in range(p.batch):
        seq_start, seq_end = int(cu[b]), int(cu[b + 1])
        for h in range(p.ho):
            qh = _head_index(h, p.hq, p.ho)
            kh = _head_index(h, p.hk, p.ho)
            vh = _head_index(h, p.hv, p.ho)
            if initial_state is None:
                state_kv = torch.zeros((128, 128), dtype=torch.float32, device=q.device)
            else:
                # Public state is [V,K]; the recurrence consumes [K,V].
                state_kv = initial_state[b, h].transpose(0, 1).to(torch.float32).clone()
            for local_t, packed_t in enumerate(range(seq_start, seq_end)):
                if p.layout == "BNSD":
                    qrow = q[b, qh, local_t].float()
                    krow = k[b, kh, local_t].float()
                    vrow = v[b, vh, local_t].float()
                    alpha = gate[b, h, local_t]
                    update = beta[b, h, local_t]
                else:
                    qrow = q[packed_t, qh].float()
                    krow = k[packed_t, kh].float()
                    vrow = v[packed_t, vh].float()
                    alpha = gate[packed_t, h]
                    update = beta[packed_t, h]
                state_kv.mul_(alpha)
                residual = (vrow - krow @ state_kv) * update
                state_kv.add_(krow[:, None] * residual[None, :])
                orow = (qrow @ state_kv) * score_scale
                if p.layout == "BNSD":
                    out[b, h, local_t] = orow.to(torch.float16)
                else:
                    out[packed_t, h] = orow.to(torch.float16)
            final_state[b, h] = state_kv.transpose(0, 1).to(torch.float16)
    return out, final_state
