# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_26poi_fused_cat_26(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7,
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index_within_block = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index_mod_256 = index_within_block % 256
    index_div_256 = index_within_block // 256
    global_index = index_within_block
    
    mod_256_mask = index_mod_256
    zero_mask = tl.full([1], 0, tl.int64)
    sixty_four_mask = tl.full([1], 64, tl.int64)
    is_less_than_64 = mod_256_mask < sixty_four_mask
    
    load0 = tl.load(input_ptr0 + (64 * index_div_256 + index_mod_256), is_less_than_64, eviction_policy='evict_last', other=0.0)
    load1 = tl.load(input_ptr1 + index_mod_256, is_less_than_64, eviction_policy='evict_last', other=0.0)
    sum_0_1 = load0 + load1
    zero_fill_0_1 = tl.full(sum_0_1.shape, 0.0, sum_0_1.dtype)
    result_0_1 = tl.where(is_less_than_64, sum_0_1, zero_fill_0_1)
    
    one_hundred_ninety_two_mask = tl.full([1], 192, tl.int64)
    is_between_64_and_192 = (mod_256_mask >= sixty_four_mask) & (mod_256_mask < one_hundred_ninety_two_mask)
    
    load2 = tl.load(input_ptr2 + (128 * index_div_256 + (-64 + index_mod_256)), is_between_64_and_192, eviction_policy='evict_last', other=0.0)
    load3 = tl.load(input_ptr3 + (-64 + index_mod_256), is_between_64_and_192, eviction_policy='evict_last', other=0.0)
    sum_2_3 = load2 + load3
    zero_fill_2_3 = tl.full(sum_2_3.shape, 0.0, sum_2_3.dtype)
    result_2_3 = tl.where(is_between_64_and_192, sum_2_3, zero_fill_2_3)
    
    two_hundred_twenty_four_mask = tl.full([1], 224, tl.int64)
    is_between_192_and_224 = (mod_256_mask >= one_hundred_ninety_two_mask) & (mod_256_mask < two_hundred_twenty_four_mask)
    
    load4 = tl.load(input_ptr4 + (32 * index_div_256 + (-192 + index_mod_256)), is_between_192_and_224, eviction_policy='evict_last', other=0.0)
    load5 = tl.load(input_ptr5 + (-192 + index_mod_256), is_between_192_and_224, eviction_policy='evict_last', other=0.0)
    sum_4_5 = load4 + load5
    zero_fill_4_5 = tl.full(sum_4_5.shape, 0.0, sum_4_5.dtype)
    result_4_5 = tl.where(is_between_192_and_224, sum_4_5, zero_fill_4_5)
    
    is_greater_than_224 = mod_256_mask >= two_hundred_twenty_four_mask
    
    load6 = tl.load(input_ptr6 + (32 * index_div_256 + (-224 + index_mod_256)), is_greater_than_224, eviction_policy='evict_last', other=0.0)
    load7 = tl.load(input_ptr7 + (-224 + index_mod_256), is_greater_than_224, eviction_policy='evict_last', other=0.0)
    sum_6_7 = load6 + load7
    zero_fill_6_7 = tl.full(sum_6_7.shape, 0.0, sum_6_7.dtype)
    result_6_7 = tl.where(is_greater_than_224, sum_6_7, zero_fill_6_7)
    
    final_result = tl.where(is_between_192_and_224, result_4_5, result_6_7)
    final_result = tl.where(is_between_64_and_192, result_2_3, final_result)
    final_result = tl.where(is_less_than_64, result_0_1, final_result)
    
    tl.store(output_ptr0 + global_index, final_result, None)