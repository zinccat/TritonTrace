# From: 70_Gemm_Sigmoid_Scaling_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_mul_sigmoid_0(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    element_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = element_indices < num_elements
    indices = element_indices
    input_values = tl.load(input_ptr + (indices), valid_mask)
    sigmoid_values = tl.sigmoid(input_values)
    scaling_factor = 2.0
    scaled_values = sigmoid_values * scaling_factor
    result_values = scaled_values + input_values
    tl.store(output_ptr + (indices), result_values, valid_mask)