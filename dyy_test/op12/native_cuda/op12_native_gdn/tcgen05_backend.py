from __future__ import annotations

import importlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable


PINNED_FLASHINFER_COMMIT = "e77a4a0d276367895c3b50a642fd8f326c03fb72"


def _default_flashinfer_repo() -> Path:
    native_root = Path(__file__).resolve().parents[1]
    return native_root.parent / "upstreams" / "flashinfer"


@lru_cache(maxsize=1)
def get_tcgen05_entrypoint() -> Callable:
    """Return the fixed FlashInfer SM100 tcgen05 implementation.

    This is intentionally an upstream-derived parity backend, not the
    independent CUDA implementation.  The commit and import location are
    checked once so performance runs do not pay a per-call subprocess cost.
    """

    repo = Path(os.environ.get("FLASHINFER_REPO", _default_flashinfer_repo())).resolve()
    if not (repo / "flashinfer" / "gdn_prefill.py").is_file():
        raise RuntimeError(f"fixed FlashInfer source not found at {repo}")
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PINNED_FLASHINFER_COMMIT:
        raise RuntimeError(
            f"tcgen05 parity backend requires FlashInfer {PINNED_FLASHINFER_COMMIT}, got {head}"
        )

    loaded = sys.modules.get("flashinfer")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
        if repo not in loaded_file.parents:
            raise RuntimeError(
                f"a different FlashInfer is already imported from {loaded_file}; expected {repo}"
            )
    else:
        sys.path.insert(0, str(repo))
        loaded = importlib.import_module("flashinfer")
        loaded_file = Path(loaded.__file__).resolve()
        if repo not in loaded_file.parents:
            raise RuntimeError(f"unexpected FlashInfer import {loaded_file}; expected {repo}")

    return importlib.import_module("flashinfer.gdn_prefill").chunk_gated_delta_rule

