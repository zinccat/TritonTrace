# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_cat_69poi_fused_avg_pool2d_cat_69(
    input_ptr, output_ptr0, output_ptr1, output_ptr2, output_ptr3, output_ptr4, 
    output_ptr5, output_ptr6, output_ptr7, output_ptr8, output_ptr9, output_ptr10, 
    output_ptr11, output_ptr12, output_ptr13, output_ptr14, output_ptr15, 
    output_ptr16, output_ptr17, output_ptr18, output_ptr19, output_ptr20, 
    output_ptr21, output_ptr22, output_ptr23, output_ptr24, output_ptr25, 
    output_ptr26, output_ptr27, output_ptr28, output_ptr29, output_ptr30, 
    output_ptr31, output_ptr32, output_ptr33, output_ptr34, output_ptr35, 
    output_ptr36, output_ptr37, output_ptr38, output_ptr39, output_ptr40, 
    output_ptr41, num_elements, BLOCK_SIZE: tl.constexpr):

    num_elements = 501760
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < num_elements

    col_index = index % 14
    row_index = index // 14
    linear_index = index
    row_col_index = index % 50176
    batch_index = index // 50176

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
    tl.store(output_ptr1 + (row_col_index + 100352 * batch_index), scaled_sum, mask)
    tl.store(output_ptr2 + (row_col_index + 106624 * batch_index), scaled_sum, mask)
    tl.store(output_ptr3 + (row_col_index + 112896 * batch_index), scaled_sum, mask)
    tl.store(output_ptr4 + (row_col_index + 119168 * batch_index), scaled_sum, mask)
    tl.store(output_ptr5 + (row_col_index + 125440 * batch_index), scaled_sum, mask)
    tl.store(output_ptr6 + (row_col_index + 131712 * batch_index), scaled_sum, mask)
    tl.store(output_ptr7 + (row_col_index + 137984 * batch_index), scaled_sum, mask)
    tl.store(output_ptr8 + (row_col_index + 144256 * batch_index), scaled_sum, mask)
    tl.store(output_ptr9 + (row_col_index + 150528 * batch_index), scaled_sum, mask)
    tl.store(output_ptr10 + (row_col_index + 156800 * batch_index), scaled_sum, mask)
    tl.store(output_ptr11 + (row_col_index + 163072 * batch_index), scaled_sum, mask)
    tl.store(output_ptr12 + (row_col_index + 169344 * batch_index), scaled_sum, mask)
    tl.store(output_ptr13 + (row_col_index + 175616 * batch_index), scaled_sum, mask)
    tl.store(output_ptr14 + (row_col_index + 181888 * batch_index), scaled_sum, mask)
    tl.store(output_ptr15 + (row_col_index + 188160 * batch_index), scaled_sum, mask)
    tl.store(output_ptr16 + (row_col_index + 194432 * batch_index), scaled_sum, mask)
    tl.store(output_ptr17 + (row_col_index + 200704 * batch_index), scaled_sum, mask)
    tl.store(output_ptr18 + (row_col_index + 206976 * batch_index), scaled_sum, mask)
    tl.store(output_ptr19 + (row_col_index + 213248 * batch_index), scaled_sum, mask)
    tl.store(output_ptr20 + (row_col_index + 219520 * batch_index), scaled_sum, mask)
    tl.store(output_ptr21 + (row_col_index + 225792 * batch_index), scaled_sum, mask)
    tl.store(output_ptr22 + (row_col_index + 232064 * batch_index), scaled_sum, mask)
    tl.store(output_ptr23 + (row_col_index + 238336 * batch_index), scaled_sum, mask)
    tl.store(output_ptr24 + (row_col_index + 244608 * batch_index), scaled_sum, mask)
    tl.store(output_ptr25 + (row_col_index + 250880 * batch_index), scaled_sum, mask)
    tl.store(output_ptr26 + (row_col_index + 257152 * batch_index), scaled_sum, mask)
    tl.store(output_ptr27 + (row_col_index + 263424 * batch_index), scaled_sum, mask)
    tl.store(output_ptr28 + (row_col_index + 269696 * batch_index), scaled_sum, mask)
    tl.store(output_ptr29 + (row_col_index + 275968 * batch_index), scaled_sum, mask)
    tl.store(output_ptr30 + (row_col_index + 282240 * batch_index), scaled_sum, mask)
    tl.store(output_ptr31 + (row_col_index + 288512 * batch_index), scaled_sum, mask)
    tl.store(output_ptr32 + (row_col_index + 294784 * batch_index), scaled_sum, mask)
    tl.store(output_ptr33 + (row_col_index + 301056 * batch_index), scaled_sum, mask)
    tl.store(output_ptr34 + (row_col_index + 307328 * batch_index), scaled_sum, mask)
    tl.store(output_ptr35 + (row_col_index + 313600 * batch_index), scaled_sum, mask)
    tl.store(output_ptr36 + (row_col_index + 319872 * batch_index), scaled_sum, mask)
    tl.store(output_ptr37 + (row_col_index + 326144 * batch_index), scaled_sum, mask)
    tl.store(output_ptr38 + (row_col_index + 332416 * batch_index), scaled_sum, mask)
    tl.store(output_ptr39 + (row_col_index + 338688 * batch_index), scaled_sum, mask)
    tl.store(output_ptr40 + (row_col_index + 344960 * batch_index), scaled_sum, mask)
    tl.store(output_ptr41 + (row_col_index + 351232 * batch_index), scaled_sum, mask)