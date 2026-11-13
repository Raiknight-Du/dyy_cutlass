#!/usr/bin/env python3
"""Emit and enforce the frozen B200 CTA resource budget."""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from op12_gdn.kernel_contract import (
    REGISTER_BUDGET,
    REGISTER_LIMIT,
    REGISTER_TOTAL,
    SMEM_BUDGET_BYTES,
    SMEM_LIMIT_BYTES,
    SMEM_RESERVED_BYTES,
    SMEM_TOTAL_BYTES,
    THREADS_PER_CTA,
    TMEM_COLUMNS,
    TMEM_LIMIT_COLUMNS,
    TMEM_TOTAL_COLUMNS,
    assert_resource_contract,
)


def main() -> None:
    assert_resource_contract()
    print(
        json.dumps(
            {
                "threads_per_cta": THREADS_PER_CTA,
                "registers": {"parts": REGISTER_BUDGET, "total": REGISTER_TOTAL, "limit": REGISTER_LIMIT},
                "smem": {
                    "parts_bytes": SMEM_BUDGET_BYTES,
                    "total_bytes": SMEM_TOTAL_BYTES,
                    "limit_bytes": SMEM_LIMIT_BYTES,
                    "reserve_bytes": SMEM_RESERVED_BYTES,
                },
                "tmem": {"parts_columns": TMEM_COLUMNS, "total_columns": TMEM_TOTAL_COLUMNS, "limit_columns": TMEM_LIMIT_COLUMNS},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
