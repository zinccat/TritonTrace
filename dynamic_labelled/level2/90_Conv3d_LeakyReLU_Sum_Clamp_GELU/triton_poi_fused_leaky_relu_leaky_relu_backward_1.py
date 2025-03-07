# From: 90_Conv3d_LeakyReLU_Sum_Clamp_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_1poi_fused_leaky_relu_leaky_relu_backward_1(in_out_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    indices = block_indices
    output_values = tl.load(in_out_ptr + (indices), mask)
    input_values = tl.load(input_ptr + (indices), mask)
    zero = 0.0
    positive_mask = output_values > zero
    negative_slope = 0.2
    negative_values = input_values * negative_slope
    leaky_relu_values = tl.where(positive_mask, input_values, negative_values)
    tl.store(in_out_ptr + (indices), leaky_relu_values, mask)