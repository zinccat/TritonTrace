# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_50(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r2 = r_index
    x3 = x_index
    x0 = (x_index % 49)
    x1 = x_index // 49

    # Load and compute intermediate values
    in_out_val = tl.load(in_out_ptr0 + (r2 + 768 * x3), r_mask, other=0.0)
    in_val0 = tl.load(in_ptr0 + (32 * x0 + 1568 * (r2 // 32) + 37632 * x1 + ((r2 % 32))), r_mask, other=0.0)
    in_val1 = tl.load(in_ptr1 + (x0 + 49 * (r2 // 32)), r_mask, eviction_policy='evict_last', other=0.0)
    in_val2 = tl.load(in_ptr2 + (r2 + 768 * x3), r_mask, other=0.0)
    in_val3 = tl.load(in_ptr3 + (r2), r_mask, eviction_policy='evict_last', other=0.0)

    # Compute temporary values
    temp_sum1 = in_val0 + in_val1
    temp_sum2 = in_out_val + temp_sum1
    temp_sum3 = in_val2 + in_val3
    temp_result = temp_sum2 + temp_sum3

    # Broadcast and apply mask
    broadcast_result = tl.broadcast_to(temp_result, [RBLOCK])
    masked_result = tl.where(r_mask, broadcast_result, 0)

    # Compute mean
    sum_masked_result = triton_helpers.promote_to_tensor(tl.sum(masked_result, 0))
    num_elements = tl.full([1], 768, tl.int32).to(tl.float32)
    mean_result = sum_masked_result / num_elements

    # Compute variance
    variance_diff = temp_result - mean_result
    variance_squared = variance_diff * variance_diff
    broadcast_variance = tl.broadcast_to(variance_squared, [RBLOCK])
    masked_variance = tl.where(r_mask, broadcast_variance, 0)
    sum_masked_variance = triton_helpers.promote_to_tensor(tl.sum(masked_variance, 0))
    variance = (temp_result - mean_result) / 768.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)

    # Compute final results
    normalized_result = variance_diff * inv_sqrt_variance
    scale_factor = 0.0013020833333333333
    scale_result = inv_sqrt_variance * scale_factor

    # Store results
    tl.store(in_out_ptr0 + (r2 + 768 * x3), normalized_result, r_mask)
    tl.store(out_ptr2 + (x3), scale_result, None)