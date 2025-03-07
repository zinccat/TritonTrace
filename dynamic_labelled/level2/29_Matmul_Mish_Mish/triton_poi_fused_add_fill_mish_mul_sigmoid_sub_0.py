# From: 29_Matmul_Mish_Mish

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
    mish_output = tl.where(is_output_greater_than_threshold, output_value, log1p_exp_output)
    tanh_mish_output = tl.extra.cuda.libdevice.tanh(mish_output)

    mish_product = output_value * tanh_mish_output
    is_mish_product_greater_than_threshold = mish_product > threshold
    exp_mish_product = tl.math.exp(mish_product)
    log1p_exp_mish_product = tl.extra.cuda.libdevice.log1p(exp_mish_product)
    mish_tanh_output = tl.where(is_mish_product_greater_than_threshold, mish_product, log1p_exp_mish_product)
    tanh_mish_tanh_output = tl.extra.cuda.libdevice.tanh(mish_tanh_output)

    sigmoid_mish_product = tl.sigmoid(mish_product)
    mish_sigmoid_product = mish_product * sigmoid_mish_product

    mish_tanh_squared = mish_tanh_output * mish_tanh_output
    one_minus_mish_tanh_squared = 1.0 - mish_tanh_squared
    mish_tanh_correction = mish_sigmoid_product * one_minus_mish_tanh_squared
    final_mish_output = mish_tanh_output + mish_tanh_correction

    input_scaled_mish = input_value * final_mish_output

    sigmoid_output = tl.sigmoid(output_value)
    output_sigmoid_product = output_value * sigmoid_output

    tanh_output_squared = tanh_output * tanh_output
    one_minus_tanh_output_squared = 1.0 - tanh_output_squared
    sigmoid_tanh_correction = output_sigmoid_product * one_minus_tanh_output_squared
    final_tanh_output = tanh_output + sigmoid_tanh_correction

    final_result = input_scaled_mish * final_tanh_output

    tl.store(in_out_ptr0 + (x0), final_result, xmask)