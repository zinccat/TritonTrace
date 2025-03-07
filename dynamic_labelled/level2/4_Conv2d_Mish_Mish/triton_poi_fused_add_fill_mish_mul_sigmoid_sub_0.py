# From: 4_Conv2d_Mish_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_mish_mul_sigmoid_sub_0poi_fused_add_fill_mish_mul_sigmoid_sub_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_value = tl.load(in_ptr0 + (x0), xmask)
    output_value = tl.load(in_out_ptr0 + (x0), xmask)
    threshold = 20.0

    is_output_greater_than_threshold = output_value > threshold
    exp_output = tl.math.exp(output_value)
    log1p_exp_output = tl.extra.cuda.libdevice.log1p(exp_output)
    output_log1p = tl.where(is_output_greater_than_threshold, output_value, log1p_exp_output)

    tanh_output_log1p = tl.extra.cuda.libdevice.tanh(output_log1p)
    mish_output = output_value * tanh_output_log1p

    is_mish_output_greater_than_threshold = mish_output > threshold
    exp_mish_output = tl.math.exp(mish_output)
    log1p_exp_mish_output = tl.extra.cuda.libdevice.log1p(exp_mish_output)
    mish_log1p = tl.where(is_mish_output_greater_than_threshold, mish_output, log1p_exp_mish_output)

    tanh_mish_log1p = tl.extra.cuda.libdevice.tanh(mish_log1p)
    sigmoid_mish_output = tl.sigmoid(mish_output)
    mish_sigmoid_product = mish_output * sigmoid_mish_output

    mish_log1p_squared = mish_log1p * mish_log1p
    one_minus_mish_log1p_squared = 1.0 - mish_log1p_squared
    mish_sigmoid_product_adjusted = mish_sigmoid_product * one_minus_mish_log1p_squared
    mish_adjusted = mish_log1p + mish_sigmoid_product_adjusted

    input_mish_adjusted = input_value * mish_adjusted

    sigmoid_output = tl.sigmoid(output_value)
    output_sigmoid_product = output_value * sigmoid_output

    tanh_output_log1p_squared = tanh_output_log1p * tanh_output_log1p
    one_minus_tanh_output_log1p_squared = 1.0 - tanh_output_log1p_squared
    output_sigmoid_product_adjusted = output_sigmoid_product * one_minus_tanh_output_log1p_squared
    output_adjusted = tanh_output_log1p + output_sigmoid_product_adjusted

    final_result = input_mish_adjusted * output_adjusted

    tl.store(in_out_ptr0 + (x0), final_result, xmask)