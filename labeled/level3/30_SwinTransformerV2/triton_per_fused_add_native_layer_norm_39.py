# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_39(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 7840
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    # Load input data
    input_data0 = tl.load(in_ptr0 + (r1 + 192 * x0), r_mask & x_mask, other=0.0)
    input_data1 = tl.load(in_ptr1 + (r1 + 192 * x0), r_mask & x_mask, other=0.0)
    input_data2 = tl.load(in_ptr2 + (r1), r_mask, eviction_policy='evict_last', other=0.0)
    input_data3 = tl.load(in_ptr3 + (r1), r_mask, eviction_policy='evict_last', other=0.0)

    # Broadcast and compute mean
    broadcast_data = tl.broadcast_to(input_data0, [XBLOCK, RBLOCK])
    tl.where(r_mask & x_mask, broadcast_data, 0)
    broadcast_mean = tl.broadcast_to(broadcast_data, [XBLOCK, RBLOCK])
    masked_broadcast = tl.where(r_mask & x_mask, broadcast_mean, 0)
    sum_broadcast = tl.sum(masked_broadcast, 1)[:, None]
    num_elements = tl.full([XBLOCK, 1], 192, tl.int32).to(tl.float32)
    mean = sum_broadcast / num_elements

    # Compute variance
    centered_data = broadcast_data - mean
    squared_data = centered_data * centered_data
    broadcast_squared = tl.broadcast_to(squared_data, [XBLOCK, RBLOCK])
    masked_squared = tl.where(r_mask & x_mask, broadcast_squared, 0)
    sum_squared = tl.sum(masked_squared, 1)[:, None]
    variance = sum_squared / 192.0

    # Compute inverse square root of variance
    epsilon = 1e-05
    adjusted_variance = variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    # Normalize input data
    normalized_data = (centered_data * inv_sqrt_variance) * input_data2 + input_data3
    output_data = input_data1 + normalized_data

    # Store results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_sqrt_variance, x_mask)
    tl.store(out_ptr1 + (r1 + 192 * x0), output_data, r_mask & x_mask)
    tl.store(out_ptr0 + (x0), mean, x_mask)