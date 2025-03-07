# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_cat_69poi_fused_avg_pool2d_cat_69(
    input_ptr, output_ptr0, output_ptr1, output_ptr2, output_ptr3, output_ptr4, 
    output_ptr5, output_ptr6, output_ptr7, output_ptr8, output_ptr9, output_ptr10, 
    output_ptr11, output_ptr12, output_ptr13, output_ptr14, output_ptr15, 
    output_ptr16, output_ptr17, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 501760
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < num_elements

    col_index = index % 14
    row_index = index // 14
    linear_index = index
    row_col_index = index % 50176
    channel_index = index // 50176

    value0 = tl.load(input_ptr + (2 * col_index + 56 * row_index), mask, eviction_policy='evict_last')
    value1 = tl.load(input_ptr + (1 + 2 * col_index + 56 * row_index), mask, eviction_policy='evict_last')
    value3 = tl.load(input_ptr + (28 + 2 * col_index + 56 * row_index), mask, eviction_policy='evict_last')
    value5 = tl.load(input_ptr + (29 + 2 * col_index + 56 * row_index), mask, eviction_policy='evict_last')

    sum1 = value1 + value0
    sum2 = value3 + sum1
    sum3 = value5 + sum2

    scale_factor = 0.25
    scaled_sum = sum3 * scale_factor

    tl.store(output_ptr0 + (linear_index), scaled_sum, mask)
    tl.store(output_ptr1 + (row_col_index + 100352 * channel_index), scaled_sum, mask)
    tl.store(output_ptr2 + (row_col_index + 106624 * channel_index), scaled_sum, mask)
    tl.store(output_ptr3 + (row_col_index + 112896 * channel_index), scaled_sum, mask)
    tl.store(output_ptr4 + (row_col_index + 119168 * channel_index), scaled_sum, mask)
    tl.store(output_ptr5 + (row_col_index + 125440 * channel_index), scaled_sum, mask)
    tl.store(output_ptr6 + (row_col_index + 131712 * channel_index), scaled_sum, mask)
    tl.store(output_ptr7 + (row_col_index + 137984 * channel_index), scaled_sum, mask)
    tl.store(output_ptr8 + (row_col_index + 144256 * channel_index), scaled_sum, mask)
    tl.store(output_ptr9 + (row_col_index + 150528 * channel_index), scaled_sum, mask)
    tl.store(output_ptr10 + (row_col_index + 156800 * channel_index), scaled_sum, mask)
    tl.store(output_ptr11 + (row_col_index + 163072 * channel_index), scaled_sum, mask)
    tl.store(output_ptr12 + (row_col_index + 169344 * channel_index), scaled_sum, mask)
    tl.store(output_ptr13 + (row_col_index + 175616 * channel_index), scaled_sum, mask)
    tl.store(output_ptr14 + (row_col_index + 181888 * channel_index), scaled_sum, mask)
    tl.store(output_ptr15 + (row_col_index + 188160 * channel_index), scaled_sum, mask)
    tl.store(output_ptr16 + (row_col_index + 194432 * channel_index), scaled_sum, mask)
    tl.store(output_ptr17 + (row_col_index + 200704 * channel_index), scaled_sum, mask)