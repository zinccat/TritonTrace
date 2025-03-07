# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_31poi_fused_max_pool2d_with_indices_31(input_ptr, output_ptr_value, output_ptr_index, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 940800
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < num_elements
    y2 = (index // 6720) % 14
    y1 = (index // 480) % 14
    x = index % 480
    batch = index // 6720
    flat_index = index

    y2_double = 2 * y2
    y2_valid = (y2_double >= 0) & (y2_double < 28)
    y1_double = 2 * y1
    y1_valid = (y1_double >= 0) & (y1_double < 28)
    y1_y2_valid = y2_valid & y1_valid

    value1 = tl.load(input_ptr + (-13920 + x + 960 * y1 + 26880 * batch), y1_y2_valid & mask, other=float("-inf"))
    value2 = tl.load(input_ptr + (-13440 + x + 960 * y1 + 26880 * batch), y1_y2_valid & mask, other=float("-inf"))
    max_value1 = triton_helpers.maximum(value2, value1)

    value3 = tl.load(input_ptr + (-12960 + x + 960 * y1 + 26880 * batch), y1_y2_valid & mask, other=float("-inf"))
    max_value2 = triton_helpers.maximum(value3, max_value1)

    y1_y2_shifted_valid = (y2_double + 1 >= 0) & (y2_double + 1 < 28) & y1_valid
    value4 = tl.load(input_ptr + (-480 + x + 960 * y1 + 26880 * batch), y1_y2_shifted_valid & mask, other=float("-inf"))
    max_value3 = triton_helpers.maximum(value4, max_value2)

    value5 = tl.load(input_ptr + (x + 960 * y1 + 26880 * batch), y1_y2_shifted_valid & mask, other=float("-inf"))
    max_value4 = triton_helpers.maximum(value5, max_value3)

    value6 = tl.load(input_ptr + (480 + x + 960 * y1 + 26880 * batch), y1_y2_shifted_valid & mask, other=float("-inf"))
    max_value5 = triton_helpers.maximum(value6, max_value4)

    y2_shifted_valid = (y2_double + 2 >= 0) & (y2_double + 2 < 28) & y1_valid
    value7 = tl.load(input_ptr + (12960 + x + 960 * y1 + 26880 * batch), y2_shifted_valid & mask, other=float("-inf"))
    max_value6 = triton_helpers.maximum(value7, max_value5)

    value8 = tl.load(input_ptr + (13440 + x + 960 * y1 + 26880 * batch), y2_shifted_valid & mask, other=float("-inf"))
    max_value7 = triton_helpers.maximum(value8, max_value6)

    value9 = tl.load(input_ptr + (13920 + x + 960 * y1 + 26880 * batch), y2_shifted_valid & mask, other=float("-inf"))
    max_value8 = triton_helpers.maximum(value9, max_value7)

    index1 = value2 > value1
    index2 = tl.full([1], 1, tl.int8)
    index3 = tl.full([1], 0, tl.int8)
    index_value = tl.where(index1, index2, index3)

    index4 = value3 > max_value1
    index5 = tl.full([1], 2, tl.int8)
    index_value = tl.where(index4, index5, index_value)

    index6 = value4 > max_value2
    index7 = tl.full([1], 3, tl.int8)
    index_value = tl.where(index6, index7, index_value)

    index8 = value5 > max_value3
    index9 = tl.full([1], 4, tl.int8)
    index_value = tl.where(index8, index9, index_value)

    index10 = value6 > max_value4
    index11 = tl.full([1], 5, tl.int8)
    index_value = tl.where(index10, index11, index_value)

    index12 = value7 > max_value5
    index13 = tl.full([1], 6, tl.int8)
    index_value = tl.where(index12, index13, index_value)

    index14 = value8 > max_value6
    index15 = tl.full([1], 7, tl.int8)
    index_value = tl.where(index14, index15, index_value)

    index16 = value9 > max_value7
    index17 = tl.full([1], 8, tl.int8)
    index_value = tl.where(index16, index17, index_value)

    tl.store(output_ptr_value + (flat_index), max_value8, mask)
    tl.store(output_ptr_index + (flat_index), index_value, mask)