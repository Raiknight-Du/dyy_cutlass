from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Torch resolves CUDA_HOME when cpp_extension is imported, so pin it before
# importing the loader.  OP12_CUDA_HOME is the explicit escape hatch.
os.environ["CUDA_HOME"] = os.environ.get("OP12_CUDA_HOME", "/usr/local/cuda-13.0")

from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_extension():
    root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
    return load(
        name="op12_native_gdn_ext",
        sources=[
            str(root / "csrc" / "binding.cpp"),
            str(root / "csrc" / "gdn_prefill_cuda.cu"),
            str(root / "csrc" / "gdn_prefill_wmma.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
        ],
        with_cuda=True,
        verbose=os.environ.get("OP12_NATIVE_VERBOSE_BUILD", "0") == "1",
    )
