# From: 54_Conv2d_Multiply_LeakyReLU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_mul_2(
    in_out_ptr0, in_ptr0, in_ptr1, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    channel_index = (index // kernel_size) % 16

    grad_output = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_tensor = tl.load(in_ptr0 + (channel_index), mask, eviction_policy='evict_last')
    grad_input = tl.load(in_ptr1 + (linear_index), mask, eviction_policy='evict_last')

    elementwise_product = grad_output * input_tensor
    zero = 0.0
    is_positive = elementwise_product > zero
    leaky_relu_slope = 0.01
    leaky_relu_result = elementwise_product * leaky_relu_slope
    leaky_relu = tl.where(is_positive, elementwise_product, leaky_relu_result)

    gelu_coefficient = 0.7071067811865476
    scaled_input = leaky_relu * gelu_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(scaled_input)
    one = 1.0
    erf_plus_one = erf_result + one
    half = 0.5
    erf_scaled = erf_plus_one * half

    squared_input = leaky_relu * leaky_relu
    negative_half = -0.5
    exp_argument = squared_input * negative_half
    exp_result = tl.math.exp(exp_argument)
    gaussian_coefficient = 0.3989422804014327
    gaussian_term = exp_result * gaussian_coefficient
    gaussian_scaled = leaky_relu * gaussian_term

    gelu_result = erf_scaled + gaussian_scaled
    elementwise_gelu = grad_input * gelu_result
    scaled_gelu = elementwise_gelu * leaky_relu_slope
    final_result = tl.where(is_positive, elementwise_gelu, scaled_gelu)
    output = final_result * input_tensor

    tl.store(in_out_ptr0 + (linear_index), output, mask)