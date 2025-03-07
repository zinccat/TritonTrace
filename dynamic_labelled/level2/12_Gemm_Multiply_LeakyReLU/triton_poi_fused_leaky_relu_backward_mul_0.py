# From: 12_Gemm_Multiply_LeakyReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_leaky_relu_backward_mul_0poi_fused_leaky_relu_backward_mul_0(
    input_grad_ptr, input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    indices = block_indices

    input_grad = tl.load(input_grad_ptr + (indices), mask).to(tl.int1)
    input_data = tl.load(input_ptr + (indices), mask)
    negative_slope = 0.1
    negative_part = input_data * negative_slope
    leaky_relu_grad = tl.where(input_grad, input_data, negative_part)
    scale_factor = 2.0
    scaled_output = leaky_relu_grad * scale_factor

    tl.store(output_ptr + (indices), scaled_output, mask)