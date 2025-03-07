# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_47poi_fused_cat_47(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7,
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 1630720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_mod_832 = xindex % 832
    x_div_832 = xindex // 832
    x_full_index = xindex

    # Temporary variables for operations
    zero_value = tl.full([1], 0, tl.int64)
    threshold_256 = tl.full([1], 256, tl.int64)
    threshold_576 = tl.full([1], 576, tl.int64)
    threshold_704 = tl.full([1], 704, tl.int64)
    threshold_832 = tl.full([1], 832, tl.int64)

    # Load and compute for first segment
    mask_256 = x_mod_832 < threshold_256
    value0 = tl.load(input_ptr0 + (256 * x_div_832 + x_mod_832), mask_256 & xmask, eviction_policy='evict_last', other=0.0)
    value1 = tl.load(input_ptr1 + x_mod_832, mask_256 & xmask, eviction_policy='evict_last', other=0.0)
    result0 = value0 + value1
    result0_filled = tl.full(result0.shape, 0.0, result0.dtype)
    segment0 = tl.where(mask_256, result0, result0_filled)

    # Load and compute for second segment
    mask_576 = (x_mod_832 >= threshold_256) & (x_mod_832 < threshold_576)
    value2 = tl.load(input_ptr2 + (320 * x_div_832 + (-256 + x_mod_832)), mask_576 & xmask, eviction_policy='evict_last', other=0.0)
    value3 = tl.load(input_ptr3 + (-256 + x_mod_832), mask_576 & xmask, eviction_policy='evict_last', other=0.0)
    result1 = value2 + value3
    result1_filled = tl.full(result1.shape, 0.0, result1.dtype)
    segment1 = tl.where(mask_576, result1, result1_filled)

    # Load and compute for third segment
    mask_704 = (x_mod_832 >= threshold_576) & (x_mod_832 < threshold_704)
    value4 = tl.load(input_ptr4 + (128 * x_div_832 + (-576 + x_mod_832)), mask_704 & xmask, eviction_policy='evict_last', other=0.0)
    value5 = tl.load(input_ptr5 + (-576 + x_mod_832), mask_704 & xmask, eviction_policy='evict_last', other=0.0)
    result2 = value4 + value5
    result2_filled = tl.full(result2.shape, 0.0, result2.dtype)
    segment2 = tl.where(mask_704, result2, result2_filled)

    # Load and compute for fourth segment
    mask_832 = x_mod_832 >= threshold_704
    value6 = tl.load(input_ptr6 + (128 * x_div_832 + (-704 + x_mod_832)), mask_832 & xmask, eviction_policy='evict_last', other=0.0)
    value7 = tl.load(input_ptr7 + (-704 + x_mod_832), mask_832 & xmask, eviction_policy='evict_last', other=0.0)
    result3 = value6 + value7
    result3_filled = tl.full(result3.shape, 0.0, result3.dtype)
    segment3 = tl.where(mask_832, result3, result3_filled)

    # Combine segments
    combined_segment2_3 = tl.where(mask_704, segment2, segment3)
    combined_segment1_23 = tl.where(mask_576, segment1, combined_segment2_3)
    final_result = tl.where(mask_256, segment0, combined_segment1_23)

    # Store the result
    tl.store(output_ptr0 + x_full_index, final_result, xmask)