# From: 20_LeakyReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_leaky_relu_0(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    indices = block_indices
    input_values = tl.load(input_ptr + (indices), mask)
    zero = 0.0
    positive_mask = input_values > zero
    negative_slope = 0.01
    negative_values = input_values * negative_slope
    output_values = tl.where(positive_mask, input_values, negative_values)
    tl.store(output_ptr + (indices), output_values, mask)