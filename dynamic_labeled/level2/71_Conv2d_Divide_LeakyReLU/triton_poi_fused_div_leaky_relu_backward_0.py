# From: 71_Conv2d_Divide_LeakyReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_leaky_relu_backward_0(input_grad_ptr, input_ptr, output_grad_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x_indices = x_index

    input_grad = tl.load(input_grad_ptr + (x_indices), x_mask).to(tl.int1)
    input_data = tl.load(input_ptr + (x_indices), x_mask)
    negative_slope = 0.01
    negative_part = input_data * negative_slope
    leaky_relu_grad = tl.where(input_grad, input_data, negative_part)

    kernel_size_float = kernel_size.to(tl.float32)
    output_grad = leaky_relu_grad / kernel_size_float

    tl.store(output_grad_ptr + (x_indices), output_grad, x_mask)