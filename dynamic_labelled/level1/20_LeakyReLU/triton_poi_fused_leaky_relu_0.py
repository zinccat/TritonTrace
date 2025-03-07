# From: 20_LeakyReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_leaky_relu_0poi_fused_leaky_relu_0(in_ptr0, out_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    input_indices = indices
    input_values = tl.load(in_ptr0 + (input_indices), mask)
    zero_threshold = 0.0
    is_positive = input_values > zero_threshold
    negative_slope = 0.01
    negative_values = input_values * negative_slope
    output_values = tl.where(is_positive, input_values, negative_values)
    tl.store(out_ptr0 + (input_indices), output_values, mask)