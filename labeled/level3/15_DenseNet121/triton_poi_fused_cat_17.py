# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_17poi_fused_cat_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    channel_index = (x_index // 3136) % 160
    spatial_index = x_index % 3136
    batch_index = x_index // 501760
    linear_index = x_index
    
    channel_value = channel_index
    zero_value = tl.full([1], 0, tl.int64)
    threshold_64 = tl.full([1], 64, tl.int64)
    threshold_96 = tl.full([1], 96, tl.int64)
    threshold_128 = tl.full([1], 128, tl.int64)
    
    condition_64 = channel_value < threshold_64
    value_0 = tl.load(in_ptr0 + (spatial_index + 3136 * channel_value + 200704 * batch_index), condition_64, other=0.0)
    
    condition_96 = channel_value >= threshold_64
    condition_96_and_96 = condition_96 & (channel_value < threshold_96)
    value_1 = tl.load(in_ptr1 + (spatial_index + 3136 * ((-64) + channel_value) + 100352 * batch_index), condition_96_and_96, other=0.0)
    
    condition_128 = channel_value >= threshold_96
    condition_128_and_128 = condition_128 & (channel_value < threshold_128)
    value_2 = tl.load(in_ptr2 + (spatial_index + 3136 * ((-96) + channel_value) + 100352 * batch_index), condition_128_and_128, other=0.0)
    
    condition_160 = channel_value >= threshold_128
    value_3 = tl.load(in_ptr3 + (spatial_index + 3136 * ((-128) + channel_value) + 100352 * batch_index), condition_160, other=0.0)
    
    value_2_or_3 = tl.where(condition_128_and_128, value_2, value_3)
    value_1_or_2_or_3 = tl.where(condition_96_and_96, value_1, value_2_or_3)
    final_value = tl.where(condition_64, value_0, value_1_or_2_or_3)
    
    tl.store(out_ptr0 + (linear_index), final_value, None)