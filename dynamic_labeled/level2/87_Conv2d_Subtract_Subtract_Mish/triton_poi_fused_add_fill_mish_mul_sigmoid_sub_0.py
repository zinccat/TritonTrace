# From: 87_Conv2d_Subtract_Subtract_Mish

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
    subtracted_value = tl.load(in_out_ptr0 + (x0), xmask)

    half = 0.5
    subtracted_half = subtracted_value - half

    two_tenths = 0.2
    subtracted_two_tenths = subtracted_half - two_tenths

    threshold = 20.0
    is_greater_than_threshold = subtracted_two_tenths > threshold

    exp_value = tl.math.exp(subtracted_two_tenths)
    log1p_exp_value = tl.extra.cuda.libdevice.log1p(exp_value)

    log1p_or_original = tl.where(is_greater_than_threshold, subtracted_two_tenths, log1p_exp_value)

    tanh_value = tl.extra.cuda.libdevice.tanh(log1p_or_original)
    sigmoid_value = tl.sigmoid(subtracted_two_tenths)

    product_sigmoid = subtracted_two_tenths * sigmoid_value

    squared_tanh = tanh_value * tanh_value
    one_minus_squared_tanh = 1.0 - squared_tanh

    mish_value = product_sigmoid * one_minus_squared_tanh
    final_mish = tanh_value + mish_value

    result = input_value * final_mish

    tl.store(in_out_ptr0 + (x0), result, xmask)