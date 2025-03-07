# From: 47_Conv3d_Mish_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_mish_mul_sigmoid_sub_tanh_backward_0(
    in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_value = tl.load(in_ptr0 + (x0), xmask)
    weight_value = tl.load(in_ptr1 + (x0), xmask)
    grad_output = tl.load(in_out_ptr0 + (x0), xmask)

    weight_squared = weight_value * weight_value
    one_minus_weight_squared = 1.0 - weight_squared
    pre_activation = input_value * one_minus_weight_squared

    threshold = 20.0
    is_large = grad_output > threshold
    exp_grad_output = tl.math.exp(grad_output)
    log1p_exp_grad_output = tl.extra.cuda.libdevice.log1p(exp_grad_output)
    log_grad_output = tl.where(is_large, grad_output, log1p_exp_grad_output)

    tanh_log_grad_output = tl.extra.cuda.libdevice.tanh(log_grad_output)
    sigmoid_grad_output = tl.sigmoid(grad_output)
    grad_output_times_sigmoid = grad_output * sigmoid_grad_output

    tanh_squared = tanh_log_grad_output * tanh_log_grad_output
    one_minus_tanh_squared = 1.0 - tanh_squared
    grad_output_sigmoid_term = grad_output_times_sigmoid * one_minus_tanh_squared

    mish_derivative = tanh_log_grad_output + grad_output_sigmoid_term
    final_gradient = pre_activation * mish_derivative

    tl.store(in_out_ptr0 + (x0), final_gradient, xmask)