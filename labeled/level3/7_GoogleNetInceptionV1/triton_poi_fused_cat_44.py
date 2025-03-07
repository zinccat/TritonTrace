# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_44poi_fused_cat_44(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7,
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    num_elements = 1034880
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < num_elements
    mod_528 = index % 528
    div_528 = index // 528
    full_index = index

    zero_value = tl.full([1], 0, tl.int64)
    threshold_112 = tl.full([1], 112, tl.int64)
    less_than_112 = mod_528 < threshold_112

    load0 = tl.load(input_ptr0 + (112 * div_528 + mod_528), less_than_112 & mask, eviction_policy='evict_last', other=0.0)
    load1 = tl.load(input_ptr1 + mod_528, less_than_112 & mask, eviction_policy='evict_last', other=0.0)
    sum_0_1 = load0 + load1
    zero_sum_0_1 = tl.full(sum_0_1.shape, 0.0, sum_0_1.dtype)
    result_0_1 = tl.where(less_than_112, sum_0_1, zero_sum_0_1)

    threshold_400 = tl.full([1], 400, tl.int64)
    greater_equal_112 = mod_528 >= threshold_112
    less_than_400 = mod_528 < threshold_400
    between_112_and_400 = greater_equal_112 & less_than_400

    load2 = tl.load(input_ptr2 + (288 * div_528 + (-112 + mod_528)), between_112_and_400 & mask, eviction_policy='evict_last', other=0.0)
    load3 = tl.load(input_ptr3 + (-112 + mod_528), between_112_and_400 & mask, eviction_policy='evict_last', other=0.0)
    sum_2_3 = load2 + load3
    zero_sum_2_3 = tl.full(sum_2_3.shape, 0.0, sum_2_3.dtype)
    result_2_3 = tl.where(between_112_and_400, sum_2_3, zero_sum_2_3)

    threshold_464 = tl.full([1], 464, tl.int64)
    greater_equal_400 = mod_528 >= threshold_400
    less_than_464 = mod_528 < threshold_464
    between_400_and_464 = greater_equal_400 & less_than_464

    load4 = tl.load(input_ptr4 + (64 * div_528 + (-400 + mod_528)), between_400_and_464 & mask, eviction_policy='evict_last', other=0.0)
    load5 = tl.load(input_ptr5 + (-400 + mod_528), between_400_and_464 & mask, eviction_policy='evict_last', other=0.0)
    sum_4_5 = load4 + load5
    zero_sum_4_5 = tl.full(sum_4_5.shape, 0.0, sum_4_5.dtype)
    result_4_5 = tl.where(between_400_and_464, sum_4_5, zero_sum_4_5)

    greater_equal_464 = mod_528 >= threshold_464

    load6 = tl.load(input_ptr6 + (64 * div_528 + (-464 + mod_528)), greater_equal_464 & mask, eviction_policy='evict_last', other=0.0)
    load7 = tl.load(input_ptr7 + (-464 + mod_528), greater_equal_464 & mask, eviction_policy='evict_last', other=0.0)
    sum_6_7 = load6 + load7
    zero_sum_6_7 = tl.full(sum_6_7.shape, 0.0, sum_6_7.dtype)
    result_6_7 = tl.where(greater_equal_464, sum_6_7, zero_sum_6_7)

    result_4_5_or_6_7 = tl.where(between_400_and_464, result_4_5, result_6_7)
    result_2_3_or_4_5_or_6_7 = tl.where(between_112_and_400, result_2_3, result_4_5_or_6_7)
    final_result = tl.where(less_than_112, result_0_1, result_2_3_or_4_5_or_6_7)

    tl.store(output_ptr0 + full_index, final_result, mask)