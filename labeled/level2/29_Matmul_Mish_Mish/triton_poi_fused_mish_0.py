# From: 29_Matmul_Mish_Mish

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_mish_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_values = tl.load(in_ptr0 + (x0), xmask)
    threshold = 20.0
    is_greater_than_threshold = input_values > threshold
    exp_input = tl.math.exp(input_values)
    log1p_exp_input = tl.extra.cuda.libdevice.log1p(exp_input)
    log1p_or_input = tl.where(is_greater_than_threshold, input_values, log1p_exp_input)
    tanh_log1p_or_input = tl.extra.cuda.libdevice.tanh(log1p_or_input)
    mish_output = input_values * tanh_log1p_or_input
    is_mish_output_greater_than_threshold = mish_output > threshold
    exp_mish_output = tl.math.exp(mish_output)
    log1p_exp_mish_output = tl.extra.cuda.libdevice.log1p(exp_mish_output)
    log1p_or_mish_output = tl.where(is_mish_output_greater_than_threshold, mish_output, log1p_exp_mish_output)
    tanh_log1p_or_mish_output = tl.extra.cuda.libdevice.tanh(log1p_or_mish_output)
    final_output = mish_output * tanh_log1p_or_mish_output
    tl.store(out_ptr0 + (x0), final_output, xmask)