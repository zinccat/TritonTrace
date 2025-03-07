# From: 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_relu_sigmoid_sigmoid_backward_threshold_backward_0(
    in_out_ptr1, in_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_data = tl.load(in_ptr0 + (x0), xmask)
    grad_output = tl.load(in_out_ptr1 + (x0), xmask)

    zero_int32 = tl.full([1], 0, tl.int32)
    max_input_grad = triton_helpers.maximum(zero_int32, grad_output)

    zero_float = 0.0
    is_positive = max_input_grad > zero_float

    leaky_relu_slope = 0.01
    leaky_relu_output = tl.where(is_positive, max_input_grad, max_input_grad * leaky_relu_slope)

    gelu_coefficient = 0.5
    gelu_scaled_input = leaky_relu_output * gelu_coefficient

    erf_coefficient = 0.7071067811865476
    erf_argument = leaky_relu_output * erf_coefficient

    erf_result = tl.extra.cuda.libdevice.erf(erf_argument)
    erf_scaled_result = gelu_scaled_input * (erf_result + 1.0)

    sigmoid_input = erf_scaled_result
    sigmoid_result = tl.sigmoid(sigmoid_input)
    sigmoid_derivative = sigmoid_result * (1.0 - sigmoid_result)

    gelu_derivative = input_data * sigmoid_derivative

    gelu_correction_term = gelu_scaled_input * (0.5 + leaky_relu_output * 0.3989422804014327 * tl.math.exp(-0.5 * leaky_relu_output * leaky_relu_output))

    grad_input = gelu_derivative * gelu_correction_term
    grad_input_leaky_relu = tl.where(is_positive, grad_input, grad_input * leaky_relu_slope)

    grad_output_updated = tl.where(max_input_grad <= zero_float, zero_float, grad_input_leaky_relu)

    tl.store(in_out_ptr1 + (x0), grad_output_updated, xmask)