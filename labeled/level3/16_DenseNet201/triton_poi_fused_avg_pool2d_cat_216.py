# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_cat_216poi_fused_avg_pool2d_cat_216(
    input_ptr, output_ptr0, output_ptr1, output_ptr2, output_ptr3, output_ptr4, 
    output_ptr5, output_ptr6, output_ptr7, output_ptr8, output_ptr9, output_ptr10, 
    output_ptr11, output_ptr12, output_ptr13, output_ptr14, output_ptr15, 
    output_ptr16, output_ptr17, output_ptr18, output_ptr19, output_ptr20, 
    output_ptr21, output_ptr22, output_ptr23, output_ptr24, output_ptr25, 
    num_elements, block_size: tl.constexpr
):
    num_elements = 439040
    offset = tl.program_id(0) * block_size
    indices = offset + tl.arange(0, block_size)[:]
    mask = indices < num_elements
    col_index = indices % 7
    row_index = indices // 7
    linear_index = indices
    sub_block_index = indices % 43904
    block_index = indices // 43904

    value0 = tl.load(input_ptr + (2 * col_index + 28 * row_index), mask, eviction_policy='evict_last')
    value1 = tl.load(input_ptr + (1 + 2 * col_index + 28 * row_index), mask, eviction_policy='evict_last')
    value3 = tl.load(input_ptr + (14 + 2 * col_index + 28 * row_index), mask, eviction_policy='evict_last')
    value5 = tl.load(input_ptr + (15 + 2 * col_index + 28 * row_index), mask, eviction_policy='evict_last')

    sum1 = value1 + value0
    sum2 = value3 + sum1
    sum3 = value5 + sum2

    scale_factor = 0.25
    scaled_sum = sum3 * scale_factor

    tl.store(output_ptr0 + (linear_index), scaled_sum, mask)
    tl.store(output_ptr1 + (sub_block_index + 56448 * block_index), scaled_sum, mask)
    tl.store(output_ptr2 + (sub_block_index + 58016 * block_index), scaled_sum, mask)
    tl.store(output_ptr3 + (sub_block_index + 59584 * block_index), scaled_sum, mask)
    tl.store(output_ptr4 + (sub_block_index + 61152 * block_index), scaled_sum, mask)
    tl.store(output_ptr5 + (sub_block_index + 62720 * block_index), scaled_sum, mask)
    tl.store(output_ptr6 + (sub_block_index + 64288 * block_index), scaled_sum, mask)
    tl.store(output_ptr7 + (sub_block_index + 65856 * block_index), scaled_sum, mask)
    tl.store(output_ptr8 + (sub_block_index + 67424 * block_index), scaled_sum, mask)
    tl.store(output_ptr9 + (sub_block_index + 68992 * block_index), scaled_sum, mask)
    tl.store(output_ptr10 + (sub_block_index + 70560 * block_index), scaled_sum, mask)
    tl.store(output_ptr11 + (sub_block_index + 72128 * block_index), scaled_sum, mask)
    tl.store(output_ptr12 + (sub_block_index + 73696 * block_index), scaled_sum, mask)
    tl.store(output_ptr13 + (sub_block_index + 75264 * block_index), scaled_sum, mask)
    tl.store(output_ptr14 + (sub_block_index + 76832 * block_index), scaled_sum, mask)
    tl.store(output_ptr15 + (sub_block_index + 78400 * block_index), scaled_sum, mask)
    tl.store(output_ptr16 + (sub_block_index + 79968 * block_index), scaled_sum, mask)
    tl.store(output_ptr17 + (sub_block_index + 81536 * block_index), scaled_sum, mask)
    tl.store(output_ptr18 + (sub_block_index + 83104 * block_index), scaled_sum, mask)
    tl.store(output_ptr19 + (sub_block_index + 84672 * block_index), scaled_sum, mask)
    tl.store(output_ptr20 + (sub_block_index + 86240 * block_index), scaled_sum, mask)
    tl.store(output_ptr21 + (sub_block_index + 87808 * block_index), scaled_sum, mask)
    tl.store(output_ptr22 + (sub_block_index + 89376 * block_index), scaled_sum, mask)
    tl.store(output_ptr23 + (sub_block_index + 90944 * block_index), scaled_sum, mask)
    tl.store(output_ptr24 + (sub_block_index + 92512 * block_index), scaled_sum, mask)
    tl.store(output_ptr25 + (sub_block_index + 94080 * block_index), scaled_sum, mask)