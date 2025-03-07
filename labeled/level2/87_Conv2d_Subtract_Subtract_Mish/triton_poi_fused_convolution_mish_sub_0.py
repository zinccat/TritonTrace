# From: 87_Conv2d_Subtract_Subtract_Mish

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_mish_sub_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    input_index = xindex
    channel_index = (xindex // 900) % 16
    input_value = tl.load(in_out_ptr0 + (input_index), None)
    weight_value = tl.load(in_ptr0 + (channel_index), None, eviction_policy='evict_last')
    sum_value = input_value + weight_value
    bias_value = 0.5
    adjusted_sum = sum_value - bias_value
    threshold_value = 0.2
    thresholded_sum = adjusted_sum - threshold_value
    max_value = 20.0
    is_greater_than_max = thresholded_sum > max_value
    exp_value = tl.math.exp(thresholded_sum)
    log1p_value = tl.extra.cuda.libdevice.log1p(exp_value)
    mish_value = tl.where(is_greater_than_max, thresholded_sum, log1p_value)
    tanh_value = tl.extra.cuda.libdevice.tanh(mish_value)
    final_output = thresholded_sum * tanh_value
    tl.store(in_out_ptr0 + (input_index), sum_value, None)
    tl.store(out_ptr0 + (input_index), final_output, None)