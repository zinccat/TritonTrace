# From: 46_Conv2d_Subtract_Tanh_Subtract_AvgPool

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_sub_tanh_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = x_index
    x1 = (x_index // 900) % 16
    input_value = tl.load(in_out_ptr0 + (x3), None)
    weight_value = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    sum_value = input_value + weight_value
    bias_value = 0.5
    adjusted_sum = sum_value - bias_value
    tanh_result = tl.extra.cuda.libdevice.tanh(adjusted_sum)
    threshold_value = 0.2
    final_result = tanh_result - threshold_value
    tl.store(in_out_ptr0 + (x3), sum_value, None)
    tl.store(out_ptr0 + (x3), final_result, None)