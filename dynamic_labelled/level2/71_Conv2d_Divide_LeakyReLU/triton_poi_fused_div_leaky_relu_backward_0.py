# From: 71_Conv2d_Divide_LeakyReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_leaky_relu_backward_0poi_fused_div_leaky_relu_backward_0(input_grad_ptr, input_data_ptr, output_grad_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_indices < num_elements
    x_indices_clamped = x_indices
    input_grad = tl.load(input_grad_ptr + (x_indices_clamped), x_mask).to(tl.int1)
    input_data = tl.load(input_data_ptr + (x_indices_clamped), x_mask)
    leaky_relu_slope = 0.01
    negative_slope = input_data * leaky_relu_slope
    activated_input = tl.where(input_grad, input_data, negative_slope)
    kernel_size_float = kernel_size.to(tl.float32)
    output_grad = activated_input / kernel_size_float
    tl.store(output_grad_ptr + (x_indices_clamped), output_grad, x_mask)