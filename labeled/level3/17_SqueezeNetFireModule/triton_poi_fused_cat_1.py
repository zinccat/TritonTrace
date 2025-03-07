# From: 17_SqueezeNetFireModule

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_1poi_fused_cat_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    channel_index = (block_indices // 50176) % 128
    spatial_index = block_indices % 50176
    batch_index = block_indices // 6422528
    global_index = block_indices
    
    channel_mask = channel_index < 64
    
    input0_value = tl.load(in_ptr0 + (spatial_index + 50176 * channel_index + 3211264 * batch_index), channel_mask, other=0.0)
    input1_value = tl.load(in_ptr1 + channel_index, channel_mask, eviction_policy='evict_last', other=0.0)
    combined_value_0 = input0_value + input1_value
    
    max_value_0 = triton_helpers.maximum(0, combined_value_0)
    zero_value_0 = tl.full(max_value_0.shape, 0.0, max_value_0.dtype)
    result_0 = tl.where(channel_mask, max_value_0, zero_value_0)
    
    channel_mask_2 = channel_index >= 64
    
    input2_value = tl.load(in_ptr2 + (spatial_index + 50176 * ((-64) + channel_index) + 3211264 * batch_index), channel_mask_2, other=0.0)
    input3_value = tl.load(in_ptr3 + ((-64) + channel_index), channel_mask_2, eviction_policy='evict_last', other=0.0)
    combined_value_1 = input2_value + input3_value
    
    max_value_1 = triton_helpers.maximum(0, combined_value_1)
    zero_value_1 = tl.full(max_value_1.shape, 0.0, max_value_1.dtype)
    result_1 = tl.where(channel_mask_2, max_value_1, zero_value_1)
    
    final_result = tl.where(channel_mask, result_0, result_1)
    tl.store(out_ptr0 + global_index, final_result, None)