# From: 87_Conv2d_Subtract_Subtract_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mish_sub_1poi_fused_mish_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_value = tl.load(in_ptr0 + (x0), xmask)
    subtract_value_1 = 0.5
    subtracted_value_1 = input_value - subtract_value_1

    subtract_value_2 = 0.2
    subtracted_value_2 = subtracted_value_1 - subtract_value_2

    threshold_value = 20.0
    is_above_threshold = subtracted_value_2 > threshold_value

    exp_value = tl.math.exp(subtracted_value_2)
    log1p_value = tl.extra.cuda.libdevice.log1p(exp_value)
    log1p_or_subtracted = tl.where(is_above_threshold, subtracted_value_2, log1p_value)

    tanh_value = tl.extra.cuda.libdevice.tanh(log1p_or_subtracted)
    mish_result = subtracted_value_2 * tanh_value

    tl.store(out_ptr0 + (x0), mish_result, xmask)