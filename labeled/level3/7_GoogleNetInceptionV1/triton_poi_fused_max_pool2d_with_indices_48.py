# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_48poi_fused_max_pool2d_with_indices_48(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 407680
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < num_elements
    channel_z = (index // 5824) % 7
    channel_y = (index // 832) % 7
    channel_x = index % 832
    batch = index // 5824
    flat_index = index

    # Calculate bounds for pooling
    bound_z = (-1) + 2 * channel_z
    zero_bound = tl.full([1], 0, tl.int64)
    upper_bound = tl.full([1], 14, tl.int64)
    valid_z = (bound_z >= zero_bound) & (bound_z < upper_bound)
    bound_y = (-1) + 2 * channel_y
    valid_y = (bound_y >= zero_bound) & (bound_y < upper_bound)
    valid_yz = valid_z & valid_y

    # Load and compute max pooling
    load_offset_1 = (-12480) + channel_x + 1664 * channel_y + 23296 * batch
    value_1 = tl.load(input_ptr + load_offset_1, valid_yz & mask, other=float("-inf"))
    load_offset_2 = (-11648) + channel_x + 1664 * channel_y + 23296 * batch
    value_2 = tl.load(input_ptr + load_offset_2, valid_yz & (bound_y == 0) & mask, other=float("-inf"))
    max_1 = triton_helpers.maximum(value_2, value_1)

    load_offset_3 = (-10816) + channel_x + 1664 * channel_y + 23296 * batch
    value_3 = tl.load(input_ptr + load_offset_3, valid_yz & (bound_y == 1) & mask, other=float("-inf"))
    max_2 = triton_helpers.maximum(value_3, max_1)

    load_offset_4 = (-832) + channel_x + 1664 * channel_y + 23296 * batch
    value_4 = tl.load(input_ptr + load_offset_4, valid_yz & (bound_z == 0) & mask, other=float("-inf"))
    max_3 = triton_helpers.maximum(value_4, max_2)

    value_5 = tl.load(input_ptr + (channel_x + 1664 * channel_y + 23296 * batch), valid_yz & (bound_z == 1) & mask, other=float("-inf"))
    max_4 = triton_helpers.maximum(value_5, max_3)

    load_offset_5 = (832 + channel_x + 1664 * channel_y + 23296 * batch)
    value_6 = tl.load(input_ptr + load_offset_5, valid_yz & (bound_z == 2) & mask, other=float("-inf"))
    max_5 = triton_helpers.maximum(value_6, max_4)

    load_offset_6 = (10816 + channel_x + 1664 * channel_y + 23296 * batch)
    value_7 = tl.load(input_ptr + load_offset_6, valid_yz & (bound_z == 3) & mask, other=float("-inf"))
    max_6 = triton_helpers.maximum(value_7, max_5)

    load_offset_7 = (11648 + channel_x + 1664 * channel_y + 23296 * batch)
    value_8 = tl.load(input_ptr + load_offset_7, valid_yz & (bound_z == 4) & mask, other=float("-inf"))
    max_7 = triton_helpers.maximum(value_8, max_6)

    load_offset_8 = (12480 + channel_x + 1664 * channel_y + 23296 * batch)
    value_9 = tl.load(input_ptr + load_offset_8, valid_yz & (bound_z == 5) & mask, other=float("-inf"))
    max_8 = triton_helpers.maximum(value_9, max_7)

    # Determine indices of max values
    index_1 = tl.where(value_2 > value_1, tl.full([1], 1, tl.int8), tl.full([1], 0, tl.int8))
    index_2 = tl.where(value_3 > max_1, tl.full([1], 2, tl.int8), index_1)
    index_3 = tl.where(value_4 > max_2, tl.full([1], 3, tl.int8), index_2)
    index_4 = tl.where(value_5 > max_3, tl.full([1], 4, tl.int8), index_3)
    index_5 = tl.where(value_6 > max_4, tl.full([1], 5, tl.int8), index_4)
    index_6 = tl.where(value_7 > max_5, tl.full([1], 6, tl.int8), index_5)
    index_7 = tl.where(value_8 > max_6, tl.full([1], 7, tl.int8), index_6)
    index_8 = tl.where(value_9 > max_7, tl.full([1], 8, tl.int8), index_7)

    # Store results
    tl.store(output_ptr_max + flat_index, max_8, mask)
    tl.store(output_ptr_indices + flat_index, index_8, mask)