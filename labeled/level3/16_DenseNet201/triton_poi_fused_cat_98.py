# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_98poi_fused_cat_98(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    channel_index = (xindex // 196) % 384
    spatial_index = xindex % 196
    batch_index = xindex // 75264
    linear_index = xindex
    channel_base = channel_index
    zero_value = tl.full([1], 0, tl.int64)
    first_threshold = tl.full([1], 256, tl.int64)
    second_threshold = tl.full([1], 288, tl.int64)
    third_threshold = tl.full([1], 320, tl.int64)
    fourth_threshold = tl.full([1], 352, tl.int64)
    
    load_mask_0 = channel_base < first_threshold
    value_0 = tl.load(in_ptr0 + (spatial_index + 196 * channel_base + 50176 * batch_index), load_mask_0 & xmask, other=0.0)
    
    load_mask_1 = (channel_base >= first_threshold) & (channel_base < second_threshold)
    value_1 = tl.load(in_ptr1 + (spatial_index + 196 * ((-256) + channel_base) + 6272 * batch_index), load_mask_1 & xmask, other=0.0)
    
    load_mask_2 = (channel_base >= second_threshold) & (channel_base < third_threshold)
    value_2 = tl.load(in_ptr2 + (spatial_index + 196 * ((-288) + channel_base) + 6272 * batch_index), load_mask_2 & xmask, other=0.0)
    
    load_mask_3 = (channel_base >= third_threshold) & (channel_base < fourth_threshold)
    value_3 = tl.load(in_ptr3 + (spatial_index + 196 * ((-320) + channel_base) + 6272 * batch_index), load_mask_3 & xmask, other=0.0)
    
    load_mask_4 = channel_base >= fourth_threshold
    value_4 = tl.load(in_ptr4 + (spatial_index + 196 * ((-352) + channel_base) + 6272 * batch_index), load_mask_4 & xmask, other=0.0)
    
    merged_value_3_4 = tl.where(load_mask_3, value_3, value_4)
    merged_value_2_3_4 = tl.where(load_mask_2, value_2, merged_value_3_4)
    merged_value_1_2_3_4 = tl.where(load_mask_1, value_1, merged_value_2_3_4)
    final_value = tl.where(load_mask_0, value_0, merged_value_1_2_3_4)
    
    tl.store(out_ptr0 + (linear_index), final_value, xmask)