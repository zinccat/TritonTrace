# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_sigmoid_14poi_fused_sigmoid_14(input_output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    num_elements = 192
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    input_values = tl.load(input_output_ptr + (indices), valid_mask)
    sigmoid_values = tl.sigmoid(input_values)
    tl.store(input_output_ptr + (indices), sigmoid_values, valid_mask)