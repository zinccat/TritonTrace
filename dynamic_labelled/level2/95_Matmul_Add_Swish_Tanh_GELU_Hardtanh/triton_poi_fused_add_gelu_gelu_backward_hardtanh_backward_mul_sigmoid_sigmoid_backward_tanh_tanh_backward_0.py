# From: 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_gelu_gelu_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_tanh_tanh_backward_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    block_index = index % 512

    input_value0 = tl.load(input_ptr0 + (element_index), mask)
    input_value1 = tl.load(input_ptr1 + (block_index), mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (element_index), mask)

    sum_input = input_value0 + input_value1
    sigmoid_output = tl.sigmoid(sum_input)
    scaled_sigmoid = sigmoid_output * sum_input
    tanh_output = tl.extra.cuda.libdevice.tanh(scaled_sigmoid)

    half = 0.5
    scaled_tanh = tanh_output * half
    sqrt_half = 0.7071067811865476
    scaled_sqrt_half_tanh = tanh_output * sqrt_half
    erf_output = tl.extra.cuda.libdevice.erf(scaled_sqrt_half_tanh)
    one = 1.0
    erf_plus_one = erf_output + one

    gelu_output = scaled_tanh * erf_plus_one
    negative_one = -1.0
    is_less_than_neg_one = gelu_output <= negative_one
    is_greater_than_one = gelu_output >= one
    out_of_bounds = is_less_than_neg_one | is_greater_than_one

    zero = 0.0
    clamped_gelu = tl.where(out_of_bounds, zero, input_value2)

    erf_plus_one_times_half = erf_plus_one * half
    tanh_squared = tanh_output * tanh_output
    negative_half = -0.5
    exp_argument = tanh_squared * negative_half
    exp_output = tl.math.exp(exp_argument)
    sqrt_two_pi = 0.3989422804014327
    gaussian = exp_output * sqrt_two_pi
    tanh_times_gaussian = tanh_output * gaussian

    derivative_gelu = erf_plus_one_times_half + tanh_times_gaussian
    scaled_clamped_gelu = clamped_gelu * derivative_gelu

    one_minus_tanh_squared = one - tanh_squared
    scaled_derivative_gelu = scaled_clamped_gelu * one_minus_tanh_squared

    one_minus_sigmoid = one - sigmoid_output
    sigmoid_times_one_minus_sigmoid = sigmoid_output * one_minus_sigmoid

    gradient_input = scaled_derivative_gelu * sigmoid_times_one_minus_sigmoid
    gradient_sum_input = scaled_derivative_gelu * sum_input

    scaled_gradient_input = gradient_input * gradient_sum_input
    final_gradient = scaled_gradient_input + (scaled_derivative_gelu * sigmoid_output * one_minus_sigmoid)

    tl.store(output_ptr0 + (element_index), scaled_clamped_gelu, mask)
    tl.store(output_ptr1 + (element_index), final_gradient, mask)