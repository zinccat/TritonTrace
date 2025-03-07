# From: 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_relu_sigmoid_sigmoid_backward_threshold_backward_0poi_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_relu_sigmoid_sigmoid_backward_threshold_backward_0(
    in_out_ptr1, in_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_value = tl.load(in_ptr0 + (x0), xmask)
    grad_output = tl.load(in_out_ptr1 + (x0), xmask)

    zero_int32 = tl.full([1], 0, tl.int32)
    max_with_zero = triton_helpers.maximum(zero_int32, grad_output)

    zero_float = 0.0
    is_positive = max_with_zero > zero_float

    leaky_relu_slope = 0.01
    leaky_relu_output = tl.where(is_positive, max_with_zero, max_with_zero * leaky_relu_slope)

    gelu_coefficient = 0.5
    scaled_leaky_relu = leaky_relu_output * gelu_coefficient

    erf_coefficient = 0.7071067811865476
    erf_input = leaky_relu_output * erf_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)

    one_float = 1.0
    erf_adjusted = erf_result + one_float

    gelu_intermediate = scaled_leaky_relu * erf_adjusted
    sigmoid_input = tl.sigmoid(gelu_intermediate)

    sigmoid_derivative = one_float - sigmoid_input
    sigmoid_gradient = sigmoid_input * sigmoid_derivative

    input_gradient = input_value * sigmoid_gradient

    gelu_derivative = erf_adjusted * gelu_coefficient
    squared_leaky_relu = leaky_relu_output * leaky_relu_output

    exp_coefficient = -0.5
    exp_input = squared_leaky_relu * exp_coefficient
    exp_result = tl.math.exp(exp_input)

    normal_dist_coefficient = 0.3989422804014327
    normal_dist = exp_result * normal_dist_coefficient

    gelu_derivative_adjusted = gelu_derivative + (leaky_relu_output * normal_dist)

    input_gradient_adjusted = input_gradient * gelu_derivative_adjusted

    leaky_relu_gradient = input_gradient_adjusted * leaky_relu_slope
    final_gradient = tl.where(is_positive, input_gradient_adjusted, leaky_relu_gradient)

    is_non_positive = max_with_zero <= zero_float
    output_gradient = tl.where(is_non_positive, zero_float, final_gradient)

    tl.store(in_out_ptr1 + (x0), output_gradient, xmask)