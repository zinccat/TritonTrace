# From: 6_GoogleNetInceptionModule

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_6poi_fused_cat_6(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7,
    output_ptr0, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)

    channel_index = (block_indices // 50176) % 512
    spatial_index = block_indices % 50176
    batch_index = block_indices // 25690112
    linear_index = block_indices

    channel_condition_192 = channel_index < 192
    value_192 = tl.load(input_ptr0 + (192 * spatial_index + 9633792 * batch_index + channel_index), channel_condition_192, eviction_policy='evict_last', other=0.0)
    value_0 = tl.load(input_ptr1 + channel_index, channel_condition_192, eviction_policy='evict_last', other=0.0)
    result_192 = value_192 + value_0
    result_192_filled = tl.full(result_192.shape, 0.0, result_192.dtype)
    result_192_conditional = tl.where(channel_condition_192, result_192, result_192_filled)

    channel_condition_400 = (channel_index >= 192) & (channel_index < 400)
    value_400 = tl.load(input_ptr2 + (208 * spatial_index + 10436608 * batch_index + (channel_index - 192)), channel_condition_400, eviction_policy='evict_last', other=0.0)
    value_192_offset = tl.load(input_ptr3 + (channel_index - 192), channel_condition_400, eviction_policy='evict_last', other=0.0)
    result_400 = value_400 + value_192_offset
    result_400_filled = tl.full(result_400.shape, 0.0, result_400.dtype)
    result_400_conditional = tl.where(channel_condition_400, result_400, result_400_filled)

    channel_condition_448 = (channel_index >= 400) & (channel_index < 448)
    value_448 = tl.load(input_ptr4 + (48 * spatial_index + 2408448 * batch_index + (channel_index - 400)), channel_condition_448, eviction_policy='evict_last', other=0.0)
    value_400_offset = tl.load(input_ptr5 + (channel_index - 400), channel_condition_448, eviction_policy='evict_last', other=0.0)
    result_448 = value_448 + value_400_offset
    result_448_filled = tl.full(result_448.shape, 0.0, result_448.dtype)
    result_448_conditional = tl.where(channel_condition_448, result_448, result_448_filled)

    channel_condition_512 = channel_index >= 448
    value_512 = tl.load(input_ptr6 + (64 * spatial_index + 3211264 * batch_index + (channel_index - 448)), channel_condition_512, eviction_policy='evict_last', other=0.0)
    value_448_offset = tl.load(input_ptr7 + (channel_index - 448), channel_condition_512, eviction_policy='evict_last', other=0.0)
    result_512 = value_512 + value_448_offset
    result_512_filled = tl.full(result_512.shape, 0.0, result_512.dtype)
    result_512_conditional = tl.where(channel_condition_512, result_512, result_512_filled)

    final_result = tl.where(channel_condition_448, result_448_conditional, result_512_conditional)
    final_result = tl.where(channel_condition_400, result_400_conditional, final_result)
    final_result = tl.where(channel_condition_192, result_192_conditional, final_result)

    tl.store(output_ptr0 + linear_index, final_result, None)