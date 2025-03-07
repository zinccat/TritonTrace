# From: 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_hardtanh_backward_mish_mul_sigmoid_sub_0poi_fused_add_fill_hardtanh_backward_mish_mul_sigmoid_sub_0(
    in_out_ptr0, in_ptr0, kernel_scale, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input and output tensors
    input_value = tl.load(in_out_ptr0 + (x0), xmask)
    kernel_value = tl.load(in_ptr0 + (x0), xmask)

    # Constants
    threshold = 20.0
    half = 0.5
    negative_one = -1.0
    one = 1.0
    zero = 0.0

    # Mish activation backward pass
    is_greater_than_threshold = input_value > threshold
    exp_input = tl.math.exp(input_value)
    log1p_exp_input = tl.extra.cuda.libdevice.log1p(exp_input)
    mish_derivative = tl.where(is_greater_than_threshold, input_value, log1p_exp_input)
    tanh_mish_derivative = tl.extra.cuda.libdevice.tanh(mish_derivative)
    mish_grad = input_value * tanh_mish_derivative

    # Hardtanh backward pass
    hardtanh_grad = mish_grad + half
    is_out_of_bounds = (hardtanh_grad <= negative_one) | (hardtanh_grad >= one)
    scaled_kernel_value = kernel_value * kernel_scale.to(tl.float32)
    hardtanh_grad_clipped = tl.where(is_out_of_bounds, zero, scaled_kernel_value)

    # Sigmoid and Mish gradient calculation
    sigmoid_input = tl.sigmoid(input_value)
    sigmoid_grad = input_value * sigmoid_input
    tanh_squared = tanh_mish_derivative * tanh_mish_derivative
    mish_sigmoid_grad = sigmoid_grad * (one - tanh_squared)
    final_grad = tanh_mish_derivative + mish_sigmoid_grad

    # Combine gradients
    combined_grad = hardtanh_grad_clipped * final_grad

    # Store result
    tl.store(in_out_ptr0 + (x0), combined_grad, xmask)