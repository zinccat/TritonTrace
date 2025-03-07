# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_7poi_fused_max_pool2d_with_indices_7(input_ptr, output_ptr_value, output_ptr_index, total_elements, BLOCK_SIZE : tl.constexpr):
    total_elements = 279936
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements

    x_coord = block_indices % 96
    y_coord = (block_indices // 96) % 54
    z_coord = block_indices // 5184
    linear_index = block_indices

    input_value_0 = tl.load(input_ptr + (x_coord + 192 * y_coord + 20928 * z_coord), valid_mask)
    input_value_1 = tl.load(input_ptr + (96 + x_coord + 192 * y_coord + 20928 * z_coord), valid_mask)
    input_value_2 = tl.load(input_ptr + (192 + x_coord + 192 * y_coord + 20928 * z_coord), valid_mask)
    input_value_3 = tl.load(input_ptr + (10464 + x_coord + 192 * y_coord + 20928 * z_coord), valid_mask)
    input_value_4 = tl.load(input_ptr + (10560 + x_coord + 192 * y_coord + 20928 * z_coord), valid_mask)
    input_value_5 = tl.load(input_ptr + (10656 + x_coord + 192 * y_coord + 20928 * z_coord), valid_mask)
    input_value_6 = tl.load(input_ptr + (20928 + x_coord + 192 * y_coord + 20928 * z_coord), valid_mask)
    input_value_7 = tl.load(input_ptr + (21024 + x_coord + 192 * y_coord + 20928 * z_coord), valid_mask)
    input_value_8 = tl.load(input_ptr + (21120 + x_coord + 192 * y_coord + 20928 * z_coord), valid_mask)

    max_value_1 = triton_helpers.maximum(input_value_1, input_value_0)
    max_value_2 = triton_helpers.maximum(input_value_2, max_value_1)
    max_value_3 = triton_helpers.maximum(input_value_3, max_value_2)
    max_value_4 = triton_helpers.maximum(input_value_4, max_value_3)
    max_value_5 = triton_helpers.maximum(input_value_5, max_value_4)
    max_value_6 = triton_helpers.maximum(input_value_6, max_value_5)
    max_value_7 = triton_helpers.maximum(input_value_7, max_value_6)
    max_value_8 = triton_helpers.maximum(input_value_8, max_value_7)

    index_1 = tl.where(input_value_1 > input_value_0, tl.full([1], 1, tl.int8), tl.full([1], 0, tl.int8))
    index_2 = tl.where(input_value_2 > max_value_1, tl.full([1], 2, tl.int8), index_1)
    index_3 = tl.where(input_value_3 > max_value_2, tl.full([1], 3, tl.int8), index_2)
    index_4 = tl.where(input_value_4 > max_value_3, tl.full([1], 4, tl.int8), index_3)
    index_5 = tl.where(input_value_5 > max_value_4, tl.full([1], 5, tl.int8), index_4)
    index_6 = tl.where(input_value_6 > max_value_5, tl.full([1], 6, tl.int8), index_5)
    index_7 = tl.where(input_value_7 > max_value_6, tl.full([1], 7, tl.int8), index_6)
    index_8 = tl.where(input_value_8 > max_value_7, tl.full([1], 8, tl.int8), index_7)

    tl.store(output_ptr_value + (linear_index), max_value_8, valid_mask)
    tl.store(output_ptr_index + (linear_index), index_8, valid_mask)