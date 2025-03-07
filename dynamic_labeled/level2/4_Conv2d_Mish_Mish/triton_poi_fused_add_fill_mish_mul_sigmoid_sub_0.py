# From: 4_Conv2d_Mish_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_mish_mul_sigmoid_sub_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
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
    mish_output = tl.where(is_output_greater_than_threshold, output_value, log1p_exp_output)
    tanh_mish_output = tl.extra.cuda.libdevice.tanh(mish_output)

    mish_product = output_value * tanh_mish_output
    is_mish_product_greater_than_threshold = mish_product > threshold
    exp_mish_product = tl.math.exp(mish_product)
    log1p_exp_mish_product = tl.extra.cuda.libdevice.log1p(exp_mish_product)
    sigmoid_mish_product = tl.where(is_mish_product_greater_than_threshold, mish_product, log1p_exp_mish_product)

    tanh_sigmoid_mish_product = tl.extra.cuda.libdevice.tanh(sigmoid_mish_product)
    sigmoid_mish_product_value = tl.sigmoid(mish_product)
    mish_sigmoid_product = mish_product * sigmoid_mish_product_value

    squared_tanh_mish_output = tanh_mish_output * tanh_mish_output
    one_minus_squared_tanh = 1.0 - squared_tanh_mish_output
    mish_sigmoid_adjusted = mish_sigmoid_product * one_minus_squared_tanh
    mish_adjusted_sum = tanh_mish_output + mish_sigmoid_adjusted

    final_mish_result = input_value * (tanh_sigmoid_mish_product + mish_sigmoid_adjusted)

    sigmoid_output_value = tl.sigmoid(output_value)
    output_sigmoid_product = output_value * sigmoid_output_value
    squared_tanh = tanh_mish_output * tanh_mish_output
    one_minus_squared_tanh_output = 1.0 - squared_tanh
    sigmoid_adjusted_product = output_sigmoid_product * one_minus_squared_tanh_output
    sigmoid_adjusted_sum = tanh_mish_output + sigmoid_adjusted_product

    final_result = final_mish_result * sigmoid_adjusted_sum

    tl.store(in_out_ptr0 + (x0), final_result, xmask)