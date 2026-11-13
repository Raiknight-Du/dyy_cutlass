#!/usr/bin/env python3
"""Construct every frozen op12 MMA atom with CuTe DSL 4.7.0."""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cutlass
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode

from op12_gdn.kernel_contract import SEVEN_GEMMS, assert_resource_contract


def main() -> None:
    assert_resource_contract()
    rows = []
    for spec in SEVEN_GEMMS:
        atom = tcgen05.MmaF8F6F4Op(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.Float32,
            spec.instruction_shape,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM if spec.a_source == "SMEM" else tcgen05.OperandSource.TMEM,
            OperandMajorMode.K if spec.a_major == "K" else OperandMajorMode.MN,
            OperandMajorMode.K if spec.b_major == "K" else OperandMajorMode.MN,
        )
        rows.append(
            {
                "number": spec.number,
                "name": spec.name,
                "class": type(atom).__name__,
                "instruction_shape": spec.instruction_shape,
                "operand_a_dtype": atom.a_dtype.__name__,
                "operand_b_dtype": atom.b_dtype.__name__,
                "accumulator_dtype": atom.acc_dtype.__name__,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
