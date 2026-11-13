"""Experimental single-CTA CuTe fusion port for op12.

This directory preserves the upstream BSD-3-Clause header in the adapted
kernel and Apache-2.0 header in the adapter. It is not selected by the public
API until compile and numerical gates pass.
"""

from .gdn_prefill_fp8 import chunk_gated_delta_rule_sm100_fp8_experimental

__all__ = ["chunk_gated_delta_rule_sm100_fp8_experimental"]
