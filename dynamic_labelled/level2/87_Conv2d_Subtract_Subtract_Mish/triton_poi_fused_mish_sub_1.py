# From: 87_Conv2d_Subtract_Subtract_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mish_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_value = tl.load(in_ptr0 + (x0), xmask)
    half = 0.5
    subtract_half = input_value - half
    two_fifths = 0.2
    subtract_two_fifths = subtract_half - two_fifths
    threshold = 20.0
    is_greater_than_threshold = subtract_two_fifths > threshold

    exp_value = tl.math.exp(subtract_two_fifths)
    log1p_exp_value = tl.extra.cuda.libdevice.log1p(exp_value)
    log1p_or_original = tl.where(is_greater_than_threshold, subtract_two_fifths, log1p_exp_value)

    tanh_value = tl.extra.cuda.libdevice.tanh(log1p_or_original)
    mish_result = subtract_two_fifths * tanh_value

    tl.store(out_ptr0 + (x0), mish_result, xmask)