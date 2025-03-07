# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_34poi_fused_max_pool2d_with_indices_34(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 940800
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = indices < num_elements

    # Calculate x2 and x1 based on indices
    x2 = (indices // 6720) % 14
    x1 = (indices // 480) % 14
    flat_index = indices

    # Calculate bounds for x2 and x1
    x2_minus_one = x2 - 1
    lower_bound = tl.full([1], 0, tl.int64)
    upper_bound = tl.full([1], 14, tl.int64)
    x2_in_bounds = (x2_minus_one >= lower_bound) & (x2_minus_one < upper_bound)
    x1_in_bounds = ((x1 - 1) >= lower_bound) & ((x1 - 1) < upper_bound)

    # Load and compute max for each position
    max_val = tl.load(input_ptr + (-7200 + flat_index), x2_in_bounds & x1_in_bounds & valid_mask, other=float("-inf"))
    max_val = triton_helpers.maximum(tl.load(input_ptr + (-6720 + flat_index), x2_in_bounds & ((x1 - 1) >= lower_bound) & ((x1 - 1) < upper_bound) & valid_mask, other=float("-inf")), max_val)
    max_val = triton_helpers.maximum(tl.load(input_ptr + (-6240 + flat_index), x2_in_bounds & ((x1 + 1) >= lower_bound) & ((x1 + 1) < upper_bound) & valid_mask, other=float("-inf")), max_val)

    x2_in_bounds = (x2 >= lower_bound) & (x2 < upper_bound)
    max_val = triton_helpers.maximum(tl.load(input_ptr + (-480 + flat_index), x2_in_bounds & x1_in_bounds & valid_mask, other=float("-inf")), max_val)
    max_val = triton_helpers.maximum(tl.load(input_ptr + flat_index, x2_in_bounds & ((x1 - 1) >= lower_bound) & ((x1 - 1) < upper_bound) & valid_mask, other=float("-inf")), max_val)
    max_val = triton_helpers.maximum(tl.load(input_ptr + (480 + flat_index), x2_in_bounds & ((x1 + 1) >= lower_bound) & ((x1 + 1) < upper_bound) & valid_mask, other=float("-inf")), max_val)

    x2_plus_one = x2 + 1
    x2_plus_one_in_bounds = (x2_plus_one >= lower_bound) & (x2_plus_one < upper_bound)
    max_val = triton_helpers.maximum(tl.load(input_ptr + (6240 + flat_index), x2_plus_one_in_bounds & x1_in_bounds & valid_mask, other=float("-inf")), max_val)
    max_val = triton_helpers.maximum(tl.load(input_ptr + (6720 + flat_index), x2_plus_one_in_bounds & ((x1 - 1) >= lower_bound) & ((x1 - 1) < upper_bound) & valid_mask, other=float("-inf")), max_val)
    max_val = triton_helpers.maximum(tl.load(input_ptr + (7200 + flat_index), x2_plus_one_in_bounds & ((x1 + 1) >= lower_bound) & ((x1 + 1) < upper_bound) & valid_mask, other=float("-inf")), max_val)

    # Determine indices of max values
    indices = tl.full([1], 0, tl.int8)
    indices = tl.where(tl.load(input_ptr + (-6720 + flat_index), x2_in_bounds & ((x1 - 1) >= lower_bound) & ((x1 - 1) < upper_bound) & valid_mask, other=float("-inf")) > max_val, tl.full([1], 1, tl.int8), indices)
    indices = tl.where(tl.load(input_ptr + (-6240 + flat_index), x2_in_bounds & ((x1 + 1) >= lower_bound) & ((x1 + 1) < upper_bound) & valid_mask, other=float("-inf")) > max_val, tl.full([1], 2, tl.int8), indices)
    indices = tl.where(tl.load(input_ptr + (-480 + flat_index), x2_in_bounds & x1_in_bounds & valid_mask, other=float("-inf")) > max_val, tl.full([1], 3, tl.int8), indices)
    indices = tl.where(tl.load(input_ptr + flat_index, x2_in_bounds & ((x1 - 1) >= lower_bound) & ((x1 - 1) < upper_bound) & valid_mask, other=float("-inf")) > max_val, tl.full([1], 4, tl.int8), indices)
    indices = tl.where(tl.load(input_ptr + (480 + flat_index), x2_in_bounds & ((x1 + 1) >= lower_bound) & ((x1 + 1) < upper_bound) & valid_mask, other=float("-inf")) > max_val, tl.full([1], 5, tl.int8), indices)
    indices = tl.where(tl.load(input_ptr + (6240 + flat_index), x2_plus_one_in_bounds & x1_in_bounds & valid_mask, other=float("-inf")) > max_val, tl.full([1], 6, tl.int8), indices)
    indices = tl.where(tl.load(input_ptr + (6720 + flat_index), x2_plus_one_in_bounds & ((x1 - 1) >= lower_bound) & ((x1 - 1) < upper_bound) & valid_mask, other=float("-inf")) > max_val, tl.full([1], 7, tl.int8), indices)
    indices = tl.where(tl.load(input_ptr + (7200 + flat_index), x2_plus_one_in_bounds & ((x1 + 1) >= lower_bound) & ((x1 + 1) < upper_bound) & valid_mask, other=float("-inf")) > max_val, tl.full([1], 8, tl.int8), indices)

    # Store results
    tl.store(output_ptr_max + flat_index, max_val, valid_mask)
    tl.store(output_ptr_indices + flat_index, indices, valid_mask)