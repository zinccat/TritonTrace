# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_layer_norm_25(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr
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
    input_data = tl.load(in_ptr0 + (r1 + 192 * x0), rmask & x_mask, other=0.0)
    mean_data = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    variance_data = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)

    # Broadcast input data
    broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
    tl.where(r_mask & x_mask, broadcast_input, 0)

    # Calculate mean
    sum_input = tl.sum(broadcast_input, 1)[:, None]
    num_elements = tl.full([XBLOCK, 1], 192, tl.int32).to(tl.float32)
    mean = sum_input / num_elements

    # Calculate variance
    centered_input = broadcast_input - mean
    squared_input = centered_input * centered_input
    broadcast_squared = tl.broadcast_to(squared_input, [XBLOCK, RBLOCK])
    sum_squared = tl.sum(tl.where(r_mask & x_mask, broadcast_squared, 0), 1)[:, None]
    variance = sum_squared / 192.0

    # Normalize
    epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance + epsilon)
    normalized_input = (input_data - mean) * inv_std

    # Scale and shift
    scaled_input = normalized_input * mean_data
    output_data = scaled_input + variance_data

    # Store results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_std, x_mask)
    tl.store(out_ptr1 + (r1 + 192 * x0), output_data, r_mask & x_mask)
    tl.store(out_ptr0 + (x0), mean, x_mask)