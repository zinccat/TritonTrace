# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_mul_softplus_softplus_backward_tanh_tanh_backward_4poi_fused_add_mul_softplus_softplus_backward_tanh_tanh_backward_4(
    in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input values
    input_value = tl.load(in_out_ptr0 + (x0), xmask)
    input_data = tl.load(in_ptr0 + (x0), xmask)

    # Constants
    threshold = 20.0
    one = 1.0

    # Softplus backward computation
    is_large = input_data > threshold
    exp_input = tl.math.exp(input_data)
    log1p_exp_input = tl.extra.cuda.libdevice.log1p(exp_input)
    softplus_result = tl.where(is_large, input_data, log1p_exp_input)

    # Tanh backward computation
    tanh_result = tl.extra.cuda.libdevice.tanh(softplus_result)
    grad_input = input_value * tanh_result

    # Compute gradients
    scaled_input = input_data * one
    is_large_scaled = scaled_input > threshold
    tanh_squared = tanh_result * tanh_result
    one_minus_tanh_squared = one - tanh_squared

    # Intermediate gradient calculations
    grad_input_scaled = input_value * input_data
    exp_scaled_input = tl.math.exp(scaled_input)
    numerator = grad_input_scaled * one_minus_tanh_squared * exp_scaled_input
    denominator = exp_scaled_input + one
    softplus_grad = numerator / denominator

    # Final gradient computation
    final_grad = tl.where(is_large_scaled, grad_input_scaled * one_minus_tanh_squared, softplus_grad)
    output_value = grad_input + final_grad

    # Store the result
    tl.store(in_out_ptr0 + (x0), output_value, xmask)