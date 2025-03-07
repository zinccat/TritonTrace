# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_22poi_fused_max_pool2d_with_indices_22(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 1505280
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = index < num_elements
    col2 = ((index // 5376) % 28)
    col1 = ((index // 192) % 28)
    col0 = (index % 192)
    row = index // 5376
    flat_index = index

    # Calculate bounds for pooling
    col2_bound = (-1) + 2 * col2
    lower_bound = tl.full([1], 0, tl.int64)
    upper_bound = tl.full([1], 56, tl.int64)
    valid_col2 = (col2_bound >= lower_bound) & (col2_bound < upper_bound)
    col1_bound = (-1) + 2 * col1
    valid_col1 = (col1_bound >= lower_bound) & (col1_bound < upper_bound)
    valid_col1_col2 = valid_col1 & valid_col2

    # Load and compare values for pooling
    val1 = tl.load(input_ptr + ((-10944) + col0 + 384 * col1 + 21504 * row), valid_col1_col2 & valid_mask, other=float("-inf"))
    val2 = tl.load(input_ptr + ((-10752) + col0 + 384 * col1 + 21504 * row), (valid_col1_col2 & ((-1 + 2 * col1) >= lower_bound) & ((-1 + 2 * col1) < upper_bound)) & valid_mask, other=float("-inf"))
    max_val1 = triton_helpers.maximum(val2, val1)

    val3 = tl.load(input_ptr + ((-10560) + col0 + 384 * col1 + 21504 * row), (valid_col1_col2 & ((1 + 2 * col1) >= lower_bound) & ((1 + 2 * col1) < upper_bound)) & valid_mask, other=float("-inf"))
    max_val2 = triton_helpers.maximum(val3, max_val1)

    valid_col2_shifted = ((-1 + 2 * col2) >= lower_bound) & ((-1 + 2 * col2) < upper_bound)
    val4 = tl.load(input_ptr + ((-192) + col0 + 384 * col1 + 21504 * row), (valid_col2_shifted & valid_col1) & valid_mask, other=float("-inf"))
    max_val3 = triton_helpers.maximum(val4, max_val2)

    val5 = tl.load(input_ptr + (col0 + 384 * col1 + 21504 * row), (valid_col2_shifted & ((-1 + 2 * col1) >= lower_bound) & ((-1 + 2 * col1) < upper_bound)) & valid_mask, other=float("-inf"))
    max_val4 = triton_helpers.maximum(val5, max_val3)

    val6 = tl.load(input_ptr + (192 + col0 + 384 * col1 + 21504 * row), (valid_col2_shifted & ((1 + 2 * col1) >= lower_bound) & ((1 + 2 * col1) < upper_bound)) & valid_mask, other=float("-inf"))
    max_val5 = triton_helpers.maximum(val6, max_val4)

    valid_col2_shifted_1 = ((1 + 2 * col2) >= lower_bound) & ((1 + 2 * col2) < upper_bound)
    val7 = tl.load(input_ptr + (10560 + col0 + 384 * col1 + 21504 * row), (valid_col2_shifted_1 & valid_col1) & valid_mask, other=float("-inf"))
    max_val6 = triton_helpers.maximum(val7, max_val5)

    val8 = tl.load(input_ptr + (10752 + col0 + 384 * col1 + 21504 * row), (valid_col2_shifted_1 & ((-1 + 2 * col1) >= lower_bound) & ((-1 + 2 * col1) < upper_bound)) & valid_mask, other=float("-inf"))
    max_val7 = triton_helpers.maximum(val8, max_val6)

    val9 = tl.load(input_ptr + (10944 + col0 + 384 * col1 + 21504 * row), (valid_col2_shifted_1 & ((1 + 2 * col1) >= lower_bound) & ((1 + 2 * col1) < upper_bound)) & valid_mask, other=float("-inf"))
    max_val8 = triton_helpers.maximum(val9, max_val7)

    # Determine indices of max values
    index1 = tl.where(val2 > val1, tl.full([1], 1, tl.int8), tl.full([1], 0, tl.int8))
    index2 = tl.where(val3 > max_val1, tl.full([1], 2, tl.int8), index1)
    index3 = tl.where(val4 > max_val2, tl.full([1], 3, tl.int8), index2)
    index4 = tl.where(val5 > max_val3, tl.full([1], 4, tl.int8), index3)
    index5 = tl.where(val6 > max_val4, tl.full([1], 5, tl.int8), index4)
    index6 = tl.where(val7 > max_val5, tl.full([1], 6, tl.int8), index5)
    index7 = tl.where(val8 > max_val6, tl.full([1], 7, tl.int8), index6)
    max_index = tl.where(val9 > max_val7, tl.full([1], 8, tl.int8), index7)

    # Store results
    tl.store(output_ptr_max + (flat_index), max_val8, valid_mask)
    tl.store(output_ptr_indices + (flat_index), max_index, valid_mask)