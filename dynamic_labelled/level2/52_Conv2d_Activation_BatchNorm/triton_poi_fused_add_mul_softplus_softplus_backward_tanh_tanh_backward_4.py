# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_mul_softplus_softplus_backward_tanh_tanh_backward_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input and output data
    input_data = tl.load(in_out_ptr0 + (x0), xmask)
    input_ptr_data = tl.load(in_ptr0 + (x0), xmask)

    # Constants
    threshold = 20.0
    one = 1.0

    # Softplus forward
    is_large = input_ptr_data > threshold
    exp_input = tl.math.exp(input_ptr_data)
    log1p_exp_input = tl.extra.cuda.libdevice.log1p(exp_input)
    softplus_output = tl.where(is_large, input_ptr_data, log1p_exp_input)

    # Tanh backward
    tanh_output = tl.extra.cuda.libdevice.tanh(softplus_output)
    grad_input_output = input_data * tanh_output

    # Softplus backward
    scaled_input = input_ptr_data * one
    is_large_backward = scaled_input > threshold
    grad_input_tanh = input_data * input_ptr_data
    tanh_squared = tanh_output * tanh_output
    one_minus_tanh_squared = one - tanh_squared
    grad_input_softplus = grad_input_tanh * one_minus_tanh_squared
    exp_scaled_input = tl.math.exp(scaled_input)
    grad_input_softplus_scaled = grad_input_softplus * exp_scaled_input
    exp_scaled_input_plus_one = exp_scaled_input + one
    grad_input_softplus_final = grad_input_softplus_scaled / exp_scaled_input_plus_one
    grad_input_softplus_combined = tl.where(is_large_backward, grad_input_softplus, grad_input_softplus_final)

    # Combine gradients
    combined_gradient = grad_input_output + grad_input_softplus_combined

    # Store result
    tl.store(in_out_ptr0 + (x0), combined_gradient, xmask)