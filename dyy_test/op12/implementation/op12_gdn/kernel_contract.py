"""Frozen fused-kernel instruction, resource, and scheduling contract.

The executable backend is deliberately separate.  These constants are used by
the CuTe construction check and resource gate so an unfused fallback cannot be
mistaken for completion of the fused SM100 kernel.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GemmContract:
    number: int
    name: str
    instruction_shape: tuple[int, int, int]
    a_source: str
    a_major: str
    b_major: str


SEVEN_GEMMS = (
    GemmContract(1, "Wkk = K @ K.T", (64, 64, 32), "SMEM", "K", "K"),
    GemmContract(2, "Wqk = Q @ K.T", (64, 64, 32), "SMEM", "K", "K"),
    GemmContract(3, "KS = K @ state", (64, 64, 32), "TMEM", "K", "K"),
    GemmContract(4, "QS = Q @ state", (64, 64, 32), "TMEM", "K", "K"),
    GemmContract(5, "NV = Ainv @ residual", (64, 64, 32), "TMEM", "K", "K"),
    GemmContract(6, "Ointra = Wqkv @ NV", (64, 64, 32), "TMEM", "K", "K"),
    GemmContract(7, "dS = K.T @ delta", (128, 64, 32), "TMEM", "K", "MN"),
)

THREADS_PER_CTA = 384
REGISTER_BUDGET = {
    "cg0_4warps_x224": 4 * 32 * 224,
    "cg1_4warps_x240": 4 * 32 * 240,
    "other_4warps_x32": 4 * 32 * 32,
}
REGISTER_TOTAL = sum(REGISTER_BUDGET.values())
REGISTER_LIMIT = 63_488

SMEM_BUDGET_BYTES = {
    "q_2stage_e4m3": 2 * 64 * 128,
    "k_4stage_e4m3": 4 * 64 * 128,
    "v_half_3stage_e4m3": 3 * 64 * 64,
    "a_inverse_3stage_e4m3": 3 * 64 * 64,
    "wqkv_2stage_e4m3": 2 * 64 * 64,
    "o_half_2stage_fp16": 2 * 64 * 64 * 2,
    "gate_beta_5stage_fp32": 2 * 5 * 64 * 4,
    "state_half_fp16": 128 * 64 * 2,
    "state_half_e4m3": 128 * 64,
    "quant_reduce_and_inverse_scratch": 16 * 1024,
    "barrier_descriptor_dsm_metadata": 8 * 1024,
}
SMEM_TOTAL_BYTES = sum(SMEM_BUDGET_BYTES.values())
SMEM_LIMIT_BYTES = 192 * 1024
SMEM_RESERVED_BYTES = SMEM_LIMIT_BYTES - SMEM_TOTAL_BYTES

TMEM_COLUMNS = {
    "state_half": 64,
    "qs": 64,
    "cg0_double_buffer": 128,
    "cg1": 64,
    "input_double_buffer": 128,
}
TMEM_TOTAL_COLUMNS = sum(TMEM_COLUMNS.values())
TMEM_LIMIT_COLUMNS = 448

CLUSTER_SHAPE = (2, 1, 1)
DV_HALF = 64
CP_MACRO_TOKENS = 4096
CP_AUTO_CLUSTER_SLOTS = 74


def assert_resource_contract() -> None:
    assert THREADS_PER_CTA == 12 * 32
    assert REGISTER_TOTAL <= REGISTER_LIMIT
    assert SMEM_TOTAL_BYTES <= SMEM_LIMIT_BYTES
    assert SMEM_RESERVED_BYTES >= 35 * 1024
    assert TMEM_TOTAL_COLUMNS <= TMEM_LIMIT_COLUMNS
    assert len(SEVEN_GEMMS) == 7
    assert all(g.instruction_shape[2] == 32 for g in SEVEN_GEMMS)
