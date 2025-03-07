# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_layer_norm_27(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    # Load input data
    input_data = tl.load(in_ptr0 + (r1 + 768 * x0), r_mask, other=0.0)
    scale_factor = tl.load(in_ptr1 + (r1), r_mask, eviction_policy='evict_last', other=0.0)
    bias = tl.load(in_ptr2 + (r1), r_mask, eviction_policy='evict_last', other=0.0)

    # Compute mean
    broadcast_input = tl.broadcast_to(input_data, [RBLOCK])
    masked_input = tl.where(r_mask, broadcast_input, 0)
    sum_input = triton_helpers.promote_to_tensor(tl.sum(masked_input, 0))
    num_elements = tl.full([1], 768, tl.int32).to(tl.float32)
    mean = sum_input / num_elements

    # Compute variance
    centered_input = input_data - mean
    squared_input = centered_input * centered_input
    broadcast_squared = tl.broadcast_to(squared_input, [RBLOCK])
    masked_squared = tl.where(r_mask, broadcast_squared, 0)
    sum_squared = triton_helpers.promote_to_tensor(tl.sum(masked_squared, 0))
    variance = sum_squared / 768.0

    # Compute inverse standard deviation
    epsilon = 1e-05
    adjusted_variance = variance + epsilon
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    # Normalize and scale
    normalized_input = (input_data - mean) * inv_std_dev
    scaled_output = normalized_input * scale_factor + bias

    # Store results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_std_dev, None)
    tl.store(out_ptr1 + (r1 + 768 * x0), scaled_output, r_mask)
    tl.store(out_ptr0 + (x0), mean, None)