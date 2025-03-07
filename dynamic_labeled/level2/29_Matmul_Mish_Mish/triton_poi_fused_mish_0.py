# From: 29_Matmul_Mish_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mish_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_value = tl.load(in_ptr0 + (x0), xmask)
    threshold = 20.0
    is_greater_than_threshold = input_value > threshold

    exp_input = tl.math.exp(input_value)
    log1p_exp_input = tl.extra.cuda.libdevice.log1p(exp_input)
    log1p_or_input = tl.where(is_greater_than_threshold, input_value, log1p_exp_input)

    tanh_log1p_or_input = tl.extra.cuda.libdevice.tanh(log1p_or_input)
    mish_result = input_value * tanh_log1p_or_input

    is_mish_result_greater_than_threshold = mish_result > threshold
    exp_mish_result = tl.math.exp(mish_result)
    log1p_exp_mish_result = tl.extra.cuda.libdevice.log1p(exp_mish_result)
    log1p_or_mish_result = tl.where(is_mish_result_greater_than_threshold, mish_result, log1p_exp_mish_result)

    tanh_log1p_or_mish_result = tl.extra.cuda.libdevice.tanh(log1p_or_mish_result)
    final_result = mish_result * tanh_log1p_or_mish_result

    tl.store(out_ptr0 + (x0), final_result, xmask)