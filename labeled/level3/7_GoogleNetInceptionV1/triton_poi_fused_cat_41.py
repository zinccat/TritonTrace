# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_41poi_fused_cat_41(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7,
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index_within_block = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index_mod_512 = index_within_block % 512
    index_div_512 = index_within_block // 512
    global_index = index_within_block
    
    tmp_index_mod_512 = index_mod_512
    tl.full([1], 0, tl.int64)
    threshold_128 = tl.full([1], 128, tl.int64)
    condition_less_than_128 = tmp_index_mod_512 < threshold_128
    
    value_from_input0 = tl.load(input_ptr0 + (128 * index_div_512 + index_mod_512), condition_less_than_128, eviction_policy='evict_last', other=0.0)
    value_from_input1 = tl.load(input_ptr1 + index_mod_512, condition_less_than_128, eviction_policy='evict_last', other=0.0)
    sum_0_1 = value_from_input0 + value_from_input1
    
    zero_filled_shape = sum_0_1.shape
    zero_filled = tl.full(zero_filled_shape, 0.0, sum_0_1.dtype)
    result_0_1 = tl.where(condition_less_than_128, sum_0_1, zero_filled)
    
    threshold_384 = tl.full([1], 384, tl.int64)
    condition_between_128_and_384 = (tmp_index_mod_512 >= threshold_128) & (tmp_index_mod_512 < threshold_384)
    
    value_from_input2 = tl.load(input_ptr2 + (256 * index_div_512 + (-128 + index_mod_512)), condition_between_128_and_384, eviction_policy='evict_last', other=0.0)
    value_from_input3 = tl.load(input_ptr3 + (-128 + index_mod_512), condition_between_128_and_384, eviction_policy='evict_last', other=0.0)
    sum_2_3 = value_from_input2 + value_from_input3
    
    zero_filled_shape_2_3 = sum_2_3.shape
    zero_filled_2_3 = tl.full(zero_filled_shape_2_3, 0.0, sum_2_3.dtype)
    result_2_3 = tl.where(condition_between_128_and_384, sum_2_3, zero_filled_2_3)
    
    threshold_448 = tl.full([1], 448, tl.int64)
    condition_between_384_and_448 = (tmp_index_mod_512 >= threshold_384) & (tmp_index_mod_512 < threshold_448)
    
    value_from_input4 = tl.load(input_ptr4 + (64 * index_div_512 + (-384 + index_mod_512)), condition_between_384_and_448, eviction_policy='evict_last', other=0.0)
    value_from_input5 = tl.load(input_ptr5 + (-384 + index_mod_512), condition_between_384_and_448, eviction_policy='evict_last', other=0.0)
    sum_4_5 = value_from_input4 + value_from_input5
    
    zero_filled_shape_4_5 = sum_4_5.shape
    zero_filled_4_5 = tl.full(zero_filled_shape_4_5, 0.0, sum_4_5.dtype)
    result_4_5 = tl.where(condition_between_384_and_448, sum_4_5, zero_filled_4_5)
    
    condition_greater_or_equal_448 = tmp_index_mod_512 >= threshold_448
    
    value_from_input6 = tl.load(input_ptr6 + (64 * index_div_512 + (-448 + index_mod_512)), condition_greater_or_equal_448, eviction_policy='evict_last', other=0.0)
    value_from_input7 = tl.load(input_ptr7 + (-448 + index_mod_512), condition_greater_or_equal_448, eviction_policy='evict_last', other=0.0)
    sum_6_7 = value_from_input6 + value_from_input7
    
    zero_filled_shape_6_7 = sum_6_7.shape
    zero_filled_6_7 = tl.full(zero_filled_shape_6_7, 0.0, sum_6_7.dtype)
    result_6_7 = tl.where(condition_greater_or_equal_448, sum_6_7, zero_filled_6_7)
    
    result_4_5_or_6_7 = tl.where(condition_between_384_and_448, result_4_5, result_6_7)
    result_2_3_or_4_5_or_6_7 = tl.where(condition_between_128_and_384, result_2_3, result_4_5_or_6_7)
    final_result = tl.where(condition_less_than_128, result_0_1, result_2_3_or_4_5_or_6_7)
    
    tl.store(output_ptr0 + global_index, final_result, None)