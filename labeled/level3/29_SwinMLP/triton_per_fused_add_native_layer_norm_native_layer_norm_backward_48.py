# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_48(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel
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

    # Load inputs
    input0 = tl.load(in_ptr0 + (r2 + 768 * x3), r_mask, other=0.0)
    input1 = tl.load(in_ptr1 + (32 * x0 + 1568 * (r2 // 32) + 37632 * x1 + ((r2 % 32))), r_mask, other=0.0)
    input2 = tl.load(in_ptr2 + (x0 + 49 * (r2 // 32)), r_mask, eviction_policy='evict_last', other=0.0)
    in_out_value = tl.load(in_out_ptr0 + (r2 + 768 * x3), r_mask, other=0.0)
    input3 = tl.load(in_ptr3 + (r2), r_mask, eviction_policy='evict_last', other=0.0)

    # Compute intermediate values
    sum_inputs = input1 + input2
    sum_inputs_with_input0 = input0 + sum_inputs
    sum_in_out_with_input3 = in_out_value + input3
    final_sum = sum_inputs_with_input0 + sum_in_out_with_input3

    # Broadcast and apply mask
    broadcast_sum = tl.broadcast_to(final_sum, [RBLOCK])
    masked_broadcast_sum = tl.where(r_mask, broadcast_sum, 0)

    # Compute mean
    sum_masked_broadcast = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast_sum, 0))
    num_elements = tl.full([1], 768, tl.int32).to(tl.float32)
    mean = sum_masked_broadcast / num_elements

    # Compute variance
    deviation = final_sum - mean
    squared_deviation = deviation * deviation
    broadcast_squared_deviation = tl.broadcast_to(squared_deviation, [RBLOCK])
    masked_squared_deviation = tl.where(r_mask, broadcast_squared_deviation, 0)
    sum_masked_squared_deviation = triton_helpers.promote_to_tensor(tl.sum(masked_squared_deviation, 0))
    variance = (sum_masked_squared_deviation / 768.0) + 1e-05

    # Compute inverse square root
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance)

    # Compute normalized output
    normalized_output = deviation * inv_sqrt_variance
    scale_factor = inv_sqrt_variance * 0.0013020833333333333

    # Store results
    tl.store(in_out_ptr0 + (r2 + 768 * x3), final_sum, r_mask)
    tl.store(out_ptr2 + (r2 + 768 * x3), normalized_output, r_mask)
    tl.store(out_ptr3 + (x3), scale_factor, None)