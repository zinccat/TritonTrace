# From: 32_HardTanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_hardtanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    input_indices = block_indices
    input_values = tl.load(in_ptr0 + (input_indices), None)
    lower_bound = -1.0
    clamped_values = triton_helpers.maximum(input_values, lower_bound)
    upper_bound = 1.0
    hardtanh_values = triton_helpers.minimum(clamped_values, upper_bound)
    tl.store(out_ptr0 + (input_indices), hardtanh_values, None)