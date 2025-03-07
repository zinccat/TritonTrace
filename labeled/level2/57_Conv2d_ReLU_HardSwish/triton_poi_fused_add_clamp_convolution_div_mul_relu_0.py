# From: 57_Conv2d_ReLU_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_clamp_convolution_div_mul_relu_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    input_index = xindex
    channel_index = (xindex // 900) % 16
    in_out_value = tl.load(in_out_ptr0 + (input_index), None)
    input_value = tl.load(in_ptr0 + (channel_index), None, eviction_policy='evict_last')
    sum_value = in_out_value + input_value
    zero_value = tl.full([1], 0, tl.int32)
    max_value = triton_helpers.maximum(zero_value, sum_value)
    bias = 3.0
    biased_value = max_value + bias
    scale_factor = 0.16666666666666666
    scaled_value = biased_value * scale_factor
    clamp_min = 0.0
    clamped_value = triton_helpers.maximum(scaled_value, clamp_min)
    clamp_max = 1.0
    final_value = triton_helpers.minimum(clamped_value, clamp_max)
    relu_value = max_value * final_value
    tl.store(in_out_ptr0 + (input_index), sum_value, None)
    tl.store(out_ptr0 + (input_index), relu_value, None)