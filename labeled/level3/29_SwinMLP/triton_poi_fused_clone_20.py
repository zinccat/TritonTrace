# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_20poi_fused_clone_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2352000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices for accessing data
    block_row = (xindex // 32) % 49
    block_col = xindex // 9408
    within_block_row = xindex % 32
    within_block_col = (xindex // 1568) % 6
    flat_index = xindex

    # Calculate temporary indices for accessing input pointers
    temp_index_1 = (-4) + 7 * ((block_col // 5) % 5) + (block_row // 7)
    zero_mask = tl.full([1], 0, tl.int64)
    temp_index_1_valid = temp_index_1 >= zero_mask
    max_index_mask = tl.full([1], 28, tl.int64)
    temp_index_1_within_bounds = temp_index_1 < max_index_mask

    temp_index_2 = (-4) + 7 * (block_col % 5) + (block_row % 7)
    temp_index_2_valid = temp_index_2 >= zero_mask
    temp_index_2_within_bounds = temp_index_2 < max_index_mask

    valid_index_mask = temp_index_1_valid & temp_index_1_within_bounds
    valid_index_mask &= temp_index_2_valid & temp_index_2_within_bounds

    # Load data from input pointers
    input_data_0 = tl.load(
        in_ptr0 + ((-22272) + within_block_row + 32 * within_block_col + 192 * (block_row % 7) +
                   1344 * (block_col % 5) + 5376 * (block_row // 7) + 37632 * ((block_col // 5) % 5) +
                   150528 * (block_col // 25)),
        valid_index_mask & xmask, other=0.0
    )

    input_data_1 = tl.load(
        in_ptr1 + (within_block_row + 32 * within_block_col),
        valid_index_mask & xmask, eviction_policy='evict_last', other=0.0
    )

    input_data_2 = tl.load(
        in_ptr2 + (within_block_row + 32 * within_block_col),
        valid_index_mask & xmask, eviction_policy='evict_last', other=0.0
    )

    # Compute result
    result = input_data_0 * input_data_1
    result += input_data_2

    # Prepare output
    output_data = tl.full(result.shape, 0.0, result.dtype)
    output_data = tl.where(valid_index_mask, result, output_data)

    # Store result
    tl.store(out_ptr0 + (flat_index), output_data, xmask)