# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_7poi_fused_clone_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3810240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_row = (xindex // 32) % 49
    x_depth = xindex // 4704
    x_col = xindex % 32
    x_channel = (xindex // 1568) % 3
    x_linear_index = xindex

    base_index = (-4) + 7 * (((x_depth // 9) % 9)) + (x_row // 7)
    zero_mask = tl.full([1], 0, tl.int64)
    base_index_valid = base_index >= zero_mask
    max_index = tl.full([1], 56, tl.int64)
    base_index_within_bounds = base_index < max_index

    offset_index = (-4) + 7 * (x_depth % 9) + ((x_row % 7))
    offset_index_valid = offset_index >= zero_mask
    offset_index_within_bounds = offset_index < max_index

    valid_index = base_index_valid & base_index_within_bounds
    valid_offset_index = valid_index & offset_index_valid
    final_mask = valid_offset_index & offset_index_within_bounds

    input_value0 = tl.load(
        in_ptr0 + ((-21888) + x_col + 32 * x_channel + 96 * ((x_row % 7)) + 672 * ((x_depth % 9)) + 5376 * (x_row // 7) + 37632 * (((x_depth // 9) % 9)) + 301056 * (x_depth // 81)),
        final_mask & xmask,
        other=0.0
    )

    input_value1 = tl.load(
        in_ptr1 + (x_col + 32 * x_channel),
        final_mask & xmask,
        eviction_policy='evict_last',
        other=0.0
    )

    multiplied_values = input_value0 * input_value1

    input_value2 = tl.load(
        in_ptr2 + (x_col + 32 * x_channel),
        final_mask & xmask,
        eviction_policy='evict_last',
        other=0.0
    )

    result_value = multiplied_values + input_value2

    zero_filled_result = tl.full(result_value.shape, 0.0, result_value.dtype)
    final_result = tl.where(final_mask, result_value, zero_filled_result)

    tl.store(out_ptr0 + (x_linear_index), final_result, xmask)