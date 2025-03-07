# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_14poi_fused_max_pool2d_with_indices_14(input_ptr, output_ptr_value, output_ptr_index, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 186624
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < num_elements
    depth = index // 6912
    height = (index // 256) % 27
    width = index % 256
    linear_index = index

    depth_double = 2 * depth
    zero = tl.full([1], 0, tl.int64)
    depth_limit = tl.full([1], 54, tl.int64)

    depth_valid = (depth_double >= zero) & (depth_double < depth_limit)
    height_double = 2 * height
    height_valid = (height_double >= zero) & (height_double < depth_limit)
    valid_indices = depth_valid & height_valid

    value_0 = tl.load(input_ptr + (width + 512 * height + 27648 * depth), valid_indices & mask, other=float("-inf"))
    value_1 = tl.load(input_ptr + (256 + width + 512 * height + 27648 * depth), (valid_indices & (height_double >= zero) & (height_double < depth_limit)) & mask, other=float("-inf"))
    max_value_1 = triton_helpers.maximum(value_1, value_0)

    value_2 = tl.load(input_ptr + (512 + width + 512 * height + 27648 * depth), (valid_indices & (height_double + 2 >= zero) & (height_double + 2 < depth_limit)) & mask, other=float("-inf"))
    max_value_2 = triton_helpers.maximum(value_2, max_value_1)

    depth_next = 1 + 2 * depth
    depth_next_valid = (depth_next >= zero) & (depth_next < depth_limit)
    next_valid_indices = depth_next_valid & height_valid

    value_3 = tl.load(input_ptr + (13824 + width + 512 * height + 27648 * depth), next_valid_indices & mask, other=float("-inf"))
    max_value_3 = triton_helpers.maximum(value_3, max_value_2)

    value_4 = tl.load(input_ptr + (14080 + width + 512 * height + 27648 * depth), (next_valid_indices & (height_double >= zero) & (height_double < depth_limit)) & mask, other=float("-inf"))
    max_value_4 = triton_helpers.maximum(value_4, max_value_3)

    value_5 = tl.load(input_ptr + (14336 + width + 512 * height + 27648 * depth), (next_valid_indices & (height_double + 2 >= zero) & (height_double + 2 < depth_limit)) & mask, other=float("-inf"))
    max_value_5 = triton_helpers.maximum(value_5, max_value_4)

    depth_next_2 = 2 + 2 * depth
    depth_next_2_valid = (depth_next_2 >= zero) & (depth_next_2 < depth_limit)
    next_2_valid_indices = depth_next_2_valid & height_valid

    value_6 = tl.load(input_ptr + (27648 + width + 512 * height + 27648 * depth), next_2_valid_indices & mask, other=float("-inf"))
    max_value_6 = triton_helpers.maximum(value_6, max_value_5)

    value_7 = tl.load(input_ptr + (27904 + width + 512 * height + 27648 * depth), (next_2_valid_indices & (height_double >= zero) & (height_double < depth_limit)) & mask, other=float("-inf"))
    max_value_7 = triton_helpers.maximum(value_7, max_value_6)

    value_8 = tl.load(input_ptr + (28160 + width + 512 * height + 27648 * depth), (next_2_valid_indices & (height_double + 2 >= zero) & (height_double + 2 < depth_limit)) & mask, other=float("-inf"))
    max_value_8 = triton_helpers.maximum(value_8, max_value_7)

    index_1 = tl.where(value_1 > value_0, tl.full([1], 1, tl.int8), tl.full([1], 0, tl.int8))
    index_2 = tl.where(value_2 > max_value_1, tl.full([1], 2, tl.int8), index_1)
    index_3 = tl.where(value_3 > max_value_2, tl.full([1], 3, tl.int8), index_2)
    index_4 = tl.where(value_4 > max_value_3, tl.full([1], 4, tl.int8), index_3)
    index_5 = tl.where(value_5 > max_value_4, tl.full([1], 5, tl.int8), index_4)
    index_6 = tl.where(value_6 > max_value_5, tl.full([1], 6, tl.int8), index_5)
    index_7 = tl.where(value_7 > max_value_6, tl.full([1], 7, tl.int8), index_6)
    index_8 = tl.where(value_8 > max_value_7, tl.full([1], 8, tl.int8), index_7)

    tl.store(output_ptr_value + (linear_index), max_value_8, mask)
    tl.store(output_ptr_index + (linear_index), index_8, mask)