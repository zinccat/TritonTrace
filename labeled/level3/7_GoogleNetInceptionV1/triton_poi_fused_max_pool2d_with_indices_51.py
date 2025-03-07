# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_51poi_fused_max_pool2d_with_indices_51(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 407680
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements

    channel_index = (indices // 5824) % 7
    row_index = (indices // 832) % 7
    flat_index = indices

    channel_lower_bound = (-1) + channel_index
    zero_bound = tl.full([1], 0, tl.int64)
    channel_upper_bound = tl.full([1], 7, tl.int64)

    channel_valid = (channel_lower_bound >= zero_bound) & (channel_lower_bound < channel_upper_bound)
    row_valid = ((-1) + row_index >= zero_bound) & ((-1) + row_index < channel_upper_bound)
    valid_indices = channel_valid & row_valid

    max_value = tl.load(input_ptr + (-6656 + flat_index), valid_indices & mask, other=float("-inf"))
    for offset in [-5824, -4992, -832, 0, 832, 4992, 5824, 6656]:
        current_value = tl.load(input_ptr + (offset + flat_index), valid_indices & mask, other=float("-inf"))
        max_value = triton_helpers.maximum(current_value, max_value)
        valid_indices = valid_indices & ((offset // 832) != 0)

    indices_map = [
        (-1, 1, 2, 3, 4, 5, 6, 7, 8)
    ]

    index_value = 1
    for i, (offset, index) in enumerate(zip([-5824, -4992, -832, 0, 832, 4992, 5824, 6656], indices_map[0])):
        current_value = tl.load(input_ptr + (offset + flat_index), valid_indices & mask, other=float("-inf"))
        max_value = triton_helpers.maximum(current_value, max_value)
        if i > 0:
            valid_indices = valid_indices & ((offset // 832) != 0)
        if max_value == current_value:
            index_value = index

    tl.store(output_ptr_max + flat_index, max_value, mask)
    tl.store(output_ptr_indices + flat_index, tl.full([1], index_value, tl.int8), mask)