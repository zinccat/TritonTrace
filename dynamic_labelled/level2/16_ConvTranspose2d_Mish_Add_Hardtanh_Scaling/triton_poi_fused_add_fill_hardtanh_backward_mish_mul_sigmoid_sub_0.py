# From: 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_hardtanh_backward_mish_mul_sigmoid_sub_0(
    in_out_ptr0, in_ptr0, kernel_scale, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x0 = x_index

    input_value = tl.load(in_out_ptr0 + (x0), x_mask)
    input_data = tl.load(in_ptr0 + (x0), x_mask)

    threshold = 20.0
    is_greater_than_threshold = input_value > threshold
    exp_value = tl.math.exp(input_value)
    log1p_value = tl.extra.cuda.libdevice.log1p(exp_value)
    adjusted_value = tl.where(is_greater_than_threshold, input_value, log1p_value)

    tanh_value = tl.extra.cuda.libdevice.tanh(adjusted_value)
    mish_value = input_value * tanh_value

    half = 0.5
    mish_plus_half = mish_value + half

    lower_bound = -1.0
    upper_bound = 1.0
    is_out_of_bounds = (mish_plus_half <= lower_bound) | (mish_plus_half >= upper_bound)

    scaled_input = kernel_scale.to(tl.float32)
    scaled_input_data = input_data * scaled_input

    zero = 0.0
    clamped_value = tl.where(is_out_of_bounds, zero, scaled_input_data)

    sigmoid_value = tl.sigmoid(input_value)
    input_times_sigmoid = input_value * sigmoid_value

    tanh_squared = tanh_value * tanh_value
    one_minus_tanh_squared = upper_bound - tanh_squared

    mish_derivative = input_times_sigmoid * one_minus_tanh_squared
    mish_derivative_plus_tanh = tanh_value + mish_derivative

    result = clamped_value * mish_derivative_plus_tanh

    tl.store(in_out_ptr0 + (x0), result, x_mask)