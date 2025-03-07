# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_15per_fused_add_native_layer_norm_15(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 31360
    rnumel = 96
    RBLOCK: tl.constexpr = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    input_data0 = tl.load(in_ptr0 + (r1 + 96 * x0), r_mask & x_mask, other=0.0)
    input_data1 = tl.load(in_ptr1 + (r1 + 96 * x0), r_mask & x_mask, other=0.0)
    input_data2 = tl.load(in_ptr2 + (r1), r_mask, eviction_policy='evict_last', other=0.0)
    input_data3 = tl.load(in_ptr3 + (r1), r_mask, eviction_policy='evict_last', other=0.0)

    broadcast_input0 = tl.broadcast_to(input_data0, [XBLOCK, RBLOCK])
    tl.where(r_mask & x_mask, broadcast_input0, 0)

    broadcast_input1 = tl.broadcast_to(broadcast_input0, [XBLOCK, RBLOCK])
    masked_broadcast = tl.where(r_mask & x_mask, broadcast_input1, 0)
    sum_masked_broadcast = tl.sum(masked_broadcast, 1)[:, None]

    num_elements = tl.full([XBLOCK, 1], 96, tl.int32).to(tl.float32)
    mean = sum_masked_broadcast / num_elements

    centered_data = broadcast_input0 - mean
    squared_centered_data = centered_data * centered_data
    broadcast_squared = tl.broadcast_to(squared_centered_data, [XBLOCK, RBLOCK])
    masked_squared = tl.where(r_mask & x_mask, broadcast_squared, 0)
    sum_squared = tl.sum(masked_squared, 1)[:, None]

    variance = sum_squared / 96.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)

    normalized_data = (centered_data * inv_stddev) * input_data2
    output_data = normalized_data + input_data3
    final_output = input_data1 + output_data

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_stddev, x_mask)
    tl.store(out_ptr1 + (r1 + 96 * x0), final_output, r_mask & x_mask)
    tl.store(out_ptr0 + (x0), mean, x_mask)