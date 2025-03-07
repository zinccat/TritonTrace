# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_sigmoid_28poi_fused_sigmoid_28(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1728
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < xnumel
    indices = block_indices
    input_values = tl.load(in_out_ptr0 + (indices), valid_mask)
    sigmoid_values = tl.sigmoid(input_values)
    tl.store(in_out_ptr0 + (indices), sigmoid_values, valid_mask)