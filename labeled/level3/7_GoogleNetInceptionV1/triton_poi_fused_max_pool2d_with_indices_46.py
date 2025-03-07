# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_46poi_fused_max_pool2d_with_indices_46(input_ptr, output_ptr_max, output_ptr_indices, total_elements, BLOCK_SIZE : tl.constexpr):
    total_elements = 1034880
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = indices < total_elements

    # Calculate x2 and x1 based on indices
    x2 = (indices // 7392) % 14
    x1 = (indices // 528) % 14
    flat_index = indices

    # Calculate bounds for x2 and x1
    x2_valid = (x2 >= 0) & (x2 < 14)
    x1_valid = (x1 >= 0) & (x1 < 14)
    valid_indices = x2_valid & x1_valid

    # Load and compute max pooling with indices
    max_val = tl.load(input_ptr + (-7920 + flat_index), valid_indices & valid_mask, other=float("-inf"))
    max_val = triton_helpers.maximum(tl.load(input_ptr + (-7392 + flat_index), valid_indices & valid_mask, other=float("-inf")), max_val)
    max_val = triton_helpers.maximum(tl.load(input_ptr + (-6864 + flat_index), valid_indices & valid_mask, other=float("-inf")), max_val)

    max_val = triton_helpers.maximum(tl.load(input_ptr + (-528 + flat_index), valid_indices & valid_mask, other=float("-inf")), max_val)
    max_val = triton_helpers.maximum(tl.load(input_ptr + flat_index, valid_indices & valid_mask, other=float("-inf")), max_val)
    max_val = triton_helpers.maximum(tl.load(input_ptr + (528 + flat_index), valid_indices & valid_mask, other=float("-inf")), max_val)

    max_val = triton_helpers.maximum(tl.load(input_ptr + (6864 + flat_index), valid_indices & valid_mask, other=float("-inf")), max_val)
    max_val = triton_helpers.maximum(tl.load(input_ptr + (7392 + flat_index), valid_indices & valid_mask, other=float("-inf")), max_val)
    max_val = triton_helpers.maximum(tl.load(input_ptr + (7920 + flat_index), valid_indices & valid_mask, other=float("-inf")), max_val)

    # Determine the index of the maximum value
    index = tl.full([1], 0, tl.int8)
    index = tl.where(tl.load(input_ptr + (-7392 + flat_index), valid_indices & valid_mask, other=float("-inf")) > max_val, tl.full([1], 1, tl.int8), index)
    index = tl.where(tl.load(input_ptr + (-6864 + flat_index), valid_indices & valid_mask, other=float("-inf")) > max_val, tl.full([1], 2, tl.int8), index)
    index = tl.where(tl.load(input_ptr + (-528 + flat_index), valid_indices & valid_mask, other=float("-inf")) > max_val, tl.full([1], 3, tl.int8), index)
    index = tl.where(tl.load(input_ptr + flat_index, valid_indices & valid_mask, other=float("-inf")) > max_val, tl.full([1], 4, tl.int8), index)
    index = tl.where(tl.load(input_ptr + (528 + flat_index), valid_indices & valid_mask, other=float("-inf")) > max_val, tl.full([1], 5, tl.int8), index)
    index = tl.where(tl.load(input_ptr + (6864 + flat_index), valid_indices & valid_mask, other=float("-inf")) > max_val, tl.full([1], 6, tl.int8), index)
    index = tl.where(tl.load(input_ptr + (7392 + flat_index), valid_indices & valid_mask, other=float("-inf")) > max_val, tl.full([1], 7, tl.int8), index)
    index = tl.where(tl.load(input_ptr + (7920 + flat_index), valid_indices & valid_mask, other=float("-inf")) > max_val, tl.full([1], 8, tl.int8), index)

    # Store the results
    tl.store(output_ptr_max + flat_index, max_val, valid_mask)
    tl.store(output_ptr_indices + flat_index, index, valid_mask)