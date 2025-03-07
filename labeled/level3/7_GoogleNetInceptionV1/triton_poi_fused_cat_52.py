# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_52poi_fused_cat_52(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7,
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 407680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_mod_832 = xindex % 832
    x_div_832 = xindex // 832
    x_full_index = xindex

    zero_value = tl.full([1], 0, tl.int64)
    threshold_256 = tl.full([1], 256, tl.int64)
    is_less_than_256 = x_mod_832 < threshold_256

    load0 = tl.load(input_ptr0 + (256 * x_div_832 + x_mod_832), is_less_than_256 & xmask, eviction_policy='evict_last', other=0.0)
    load1 = tl.load(input_ptr1 + x_mod_832, is_less_than_256 & xmask, eviction_policy='evict_last', other=0.0)
    sum_0_1 = load0 + load1
    zero_like_sum_0_1 = tl.full(sum_0_1.shape, 0.0, sum_0_1.dtype)
    result_0_1 = tl.where(is_less_than_256, sum_0_1, zero_like_sum_0_1)

    threshold_576 = tl.full([1], 576, tl.int64)
    is_between_256_and_576 = (x_mod_832 >= threshold_256) & (x_mod_832 < threshold_576)

    load2 = tl.load(input_ptr2 + (320 * x_div_832 + (-256 + x_mod_832)), is_between_256_and_576 & xmask, eviction_policy='evict_last', other=0.0)
    load3 = tl.load(input_ptr3 + (-256 + x_mod_832), is_between_256_and_576 & xmask, eviction_policy='evict_last', other=0.0)
    sum_2_3 = load2 + load3
    zero_like_sum_2_3 = tl.full(sum_2_3.shape, 0.0, sum_2_3.dtype)
    result_2_3 = tl.where(is_between_256_and_576, sum_2_3, zero_like_sum_2_3)

    threshold_704 = tl.full([1], 704, tl.int64)
    is_between_576_and_704 = (x_mod_832 >= threshold_576) & (x_mod_832 < threshold_704)

    load4 = tl.load(input_ptr4 + (128 * x_div_832 + (-576 + x_mod_832)), is_between_576_and_704 & xmask, eviction_policy='evict_last', other=0.0)
    load5 = tl.load(input_ptr5 + (-576 + x_mod_832), is_between_576_and_704 & xmask, eviction_policy='evict_last', other=0.0)
    sum_4_5 = load4 + load5
    zero_like_sum_4_5 = tl.full(sum_4_5.shape, 0.0, sum_4_5.dtype)
    result_4_5 = tl.where(is_between_576_and_704, sum_4_5, zero_like_sum_4_5)

    is_greater_than_704 = x_mod_832 >= threshold_704

    load6 = tl.load(input_ptr6 + (128 * x_div_832 + (-704 + x_mod_832)), is_greater_than_704 & xmask, eviction_policy='evict_last', other=0.0)
    load7 = tl.load(input_ptr7 + (-704 + x_mod_832), is_greater_than_704 & xmask, eviction_policy='evict_last', other=0.0)
    sum_6_7 = load6 + load7
    zero_like_sum_6_7 = tl.full(sum_6_7.shape, 0.0, sum_6_7.dtype)
    result_6_7 = tl.where(is_greater_than_704, sum_6_7, zero_like_sum_6_7)

    result_4_5_or_6_7 = tl.where(is_between_576_and_704, result_4_5, result_6_7)
    result_2_3_or_4_5_or_6_7 = tl.where(is_between_256_and_576, result_2_3, result_4_5_or_6_7)
    final_result = tl.where(is_less_than_256, result_0_1, result_2_3_or_4_5_or_6_7)

    tl.store(output_ptr0 + x_full_index, final_result, xmask)