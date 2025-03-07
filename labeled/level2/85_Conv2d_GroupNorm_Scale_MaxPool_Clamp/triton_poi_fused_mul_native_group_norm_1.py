# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_mul_native_group_norm_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    group_index = index // 900
    channel_index = group_index % 16
    
    input_val0 = tl.load(input_ptr0 + index, None)
    input_val1 = tl.load(input_ptr1 + (group_index // 2), None, eviction_policy='evict_last')
    input_val2 = tl.load(input_ptr2 + (group_index // 2), None, eviction_policy='evict_last')
    input_val3 = tl.load(input_ptr3 + channel_index, None, eviction_policy='evict_last')
    input_val4 = tl.load(input_ptr4 + channel_index, None, eviction_policy='evict_last')
    input_val5 = tl.load(input_ptr5 + channel_index, None, eviction_policy='evict_last')
    
    normalized_val = (input_val0 - input_val1) * input_val2
    scaled_val = normalized_val * input_val3
    biased_val = scaled_val + input_val4
    final_val = biased_val * input_val5
    
    tl.store(output_ptr0 + index, final_val, None)