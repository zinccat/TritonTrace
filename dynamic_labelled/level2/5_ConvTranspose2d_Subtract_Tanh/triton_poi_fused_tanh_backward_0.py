# From: 5_ConvTranspose2d_Subtract_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_tanh_backward_0poi_fused_tanh_backward_0(input_grad_ptr, input_data_ptr, output_grad_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    indices = block_indices

    input_grad = tl.load(input_grad_ptr + (indices), mask)
    input_data = tl.load(input_data_ptr + (indices), mask)
    input_data_squared = input_data * input_data
    one = 1.0
    one_minus_input_data_squared = one - input_data_squared
    output_grad = input_grad * one_minus_input_data_squared

    tl.store(output_grad_ptr + (indices), output_grad, mask)