# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_33poi_fused_clone_33(input_ptr0, input_ptr1, input_ptr2, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 1693440
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    block_row = (block_indices // 32) % 49
    block_depth = block_indices // 18816
    element_within_block = block_indices % 32
    block_channel = (block_indices // 1568) % 12
    global_index = block_indices

    channel_index = (-4) + 7 * (((block_depth // 3) % 3)) + (block_row // 7)
    zero_mask = tl.full([1], 0, tl.int64)
    valid_channel = channel_index >= zero_mask
    max_channel = tl.full([1], 14, tl.int64)
    channel_within_bounds = channel_index < max_channel

    channel_index_mod = (-4) + 7 * (block_depth % 3) + (block_row % 7)
    valid_channel_mod = channel_index_mod >= zero_mask
    channel_mod_within_bounds = channel_index_mod < max_channel

    valid_channel_combined = valid_channel & channel_within_bounds
    valid_channel_mod_combined = valid_channel_combined & valid_channel_mod
    valid_indices = valid_channel_mod_combined & channel_mod_within_bounds

    input_value0 = tl.load(
        input_ptr0 + ((-23040) + element_within_block + 32 * block_channel + 384 * (block_row % 7) + 2688 * (block_depth % 3) + 5376 * (block_row // 7) + 37632 * (((block_depth // 3) % 3)) + 75264 * (block_depth // 9)),
        valid_indices & valid_mask,
        other=0.0
    )

    input_value1 = tl.load(
        input_ptr1 + (element_within_block + 32 * block_channel),
        valid_indices & valid_mask,
        eviction_policy='evict_last',
        other=0.0
    )

    product = input_value0 * input_value1

    input_value2 = tl.load(
        input_ptr2 + (element_within_block + 32 * block_channel),
        valid_indices & valid_mask,
        eviction_policy='evict_last',
        other=0.0
    )

    result = product + input_value2

    zero_result = tl.full(result.shape, 0.0, result.dtype)
    final_result = tl.where(valid_indices, result, zero_result)

    tl.store(output_ptr0 + (global_index), final_result, valid_mask)