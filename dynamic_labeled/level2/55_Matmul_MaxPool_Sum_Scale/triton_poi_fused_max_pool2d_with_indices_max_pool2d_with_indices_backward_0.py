# From: 55_Matmul_MaxPool_Sum_Scale

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_0(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements

    col_index = block_indices % 5
    row_index = block_indices // 5
    linear_index = block_indices

    offset_calculation = (col_index // 2) * (col_index // 2 > 0)
    max_offset = -1 + (2 * (2 <= (1 + (col_index // 2))) + (1 + (col_index // 2)) * ((1 + (col_index // 2)) < 2))
    valid_offset = offset_calculation <= max_offset
    final_offset = max_offset * (max_offset < offset_calculation)

    input_value0 = tl.load(input_ptr0 + (2 * row_index + (offset_calculation * valid_offset + final_offset)), mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (row_index), mask, eviction_policy='evict_last')

    divisor = tl.full([1], 2, tl.int32)
    quotient = tl.where((input_value0 < 0) != (divisor < 0), tl.where(input_value0 % divisor != 0, input_value0 // divisor - 1, input_value0 // divisor), input_value0 // divisor)
    product = quotient * divisor
    remainder = input_value0 - product

    index_base = tl.full([1], 0, tl.int64)
    adjusted_index = index_base + quotient
    offset_multiplier = 2 * (offset_calculation * valid_offset + final_offset)
    adjusted_offset = offset_multiplier + remainder

    stride = tl.full([1], 5, tl.int64)
    linear_index_adjusted = adjusted_index * stride + adjusted_offset

    scale_factor = 0.5
    scaled_input_value1 = input_value1 * scale_factor

    comparison_index = col_index
    condition = linear_index_adjusted == comparison_index

    output_value = tl.where(condition, scaled_input_value1, tl.full([1], 0.0, tl.float32))
    tl.store(output_ptr0 + (linear_index), output_value, mask)