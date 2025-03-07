# From: 20_LeakyReLU

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_leaky_relu_0(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    indices = block_indices
    input_values = tl.load(input_ptr + (indices), None)
    zero_threshold = 0.0
    is_positive = input_values > zero_threshold
    negative_slope = 0.01
    negative_values = input_values * negative_slope
    output_values = tl.where(is_positive, input_values, negative_values)
    tl.store(output_ptr + (indices), output_values, None)