# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_min_native_group_norm_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, output_ptr1, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 128
    rnumel = 256
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    min_value_buffer = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    min_value_index_buffer = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    min_index_buffer = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index

        input_value0 = tl.load(input_ptr0 + (r1 + (256 * x0)), rmask & x_mask, eviction_policy='evict_first', other=0.0)
        input_value1 = tl.load(input_ptr1 + ((8 * x0) + (r1 // 32)), rmask & x_mask, eviction_policy='evict_last', other=0.0)
        input_value2 = tl.load(input_ptr2 + ((8 * x0) + (r1 // 32)), rmask & x_mask, eviction_policy='evict_last', other=0.0)
        input_value3 = tl.load(input_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        input_value4 = tl.load(input_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)

        normalized_value = input_value0 - input_value1
        scale_factor = 32.0
        epsilon = 1e-05
        normalized_value2 = input_value2 / scale_factor
        adjusted_value = normalized_value2 + epsilon
        reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_value)
        scaled_value = normalized_value * reciprocal_sqrt
        weighted_value = scaled_value * input_value3
        final_value = weighted_value + input_value4

        broadcasted_value = tl.broadcast_to(final_value, [XBLOCK, RBLOCK])
        min_value_buffer = triton_helpers.minimum(min_value_buffer, broadcasted_value)
        min_value_buffer = tl.where(rmask & x_mask, min_value_buffer, min_value_buffer)

        min_value_next, min_index_next = triton_helpers.minimum_with_index(
            min_value_index_buffer, min_index_buffer, broadcasted_value, r_index
        )
        min_value_index_buffer = tl.where(rmask & x_mask, min_value_next, min_value_index_buffer)
        min_index_buffer = tl.where(rmask & x_mask, min_index_next, min_index_buffer)

    min_across_rows = triton_helpers.min2(min_value_buffer, 1)[:, None]
    tl.store(output_ptr0 + (x0), min_across_rows, x_mask)

    _, min_index_across_rows = triton_helpers.min_with_index(min_value_index_buffer, min_index_buffer, 1)
    min_index_across_rows = min_index_across_rows[:, None]
    tl.store(output_ptr1 + (x0), min_index_across_rows, x_mask)