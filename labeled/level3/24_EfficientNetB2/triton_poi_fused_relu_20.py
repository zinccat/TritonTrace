# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_relu_20poi_fused_relu_20(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 288
    block_offset = tl.program_id(0) * XBLOCK
    element_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = element_indices < xnumel
    indices = element_indices
    input_values = tl.load(in_out_ptr0 + (indices), valid_mask)
    zero_value = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_value, input_values)
    tl.store(in_out_ptr0 + (indices), relu_output, valid_mask)