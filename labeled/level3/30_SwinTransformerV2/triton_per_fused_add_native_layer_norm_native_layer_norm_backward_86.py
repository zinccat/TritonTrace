# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_86(
    in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr3, xnumel, rnumel
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

    # Load inputs with masking
    input0 = tl.load(in_ptr0 + (r1 + 768 * x0), r_mask, other=0.0)
    input1 = tl.load(in_ptr1 + (r1 + 768 * x0), r_mask, other=0.0)
    input2 = tl.load(in_ptr2 + (r1), r_mask, eviction_policy='evict_last', other=0.0)
    input3 = tl.load(in_ptr3 + (r1), r_mask, eviction_policy='evict_last', other=0.0)

    # Compute mean
    broadcast_input0 = tl.broadcast_to(input0, [RBLOCK])
    masked_broadcast_input0 = tl.where(r_mask, broadcast_input0, 0)
    sum_masked_broadcast_input0 = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast_input0, 0))
    mean = sum_masked_broadcast_input0 / 768.0

    # Compute variance
    centered_input0 = input0 - mean
    squared_centered_input0 = centered_input0 * centered_input0
    broadcast_squared_centered_input0 = tl.broadcast_to(squared_centered_input0, [RBLOCK])
    masked_broadcast_squared_centered_input0 = tl.where(r_mask, broadcast_squared_centered_input0, 0)
    sum_masked_broadcast_squared_centered_input0 = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast_squared_centered_input0, 0))
    variance = sum_masked_broadcast_squared_centered_input0 / 768.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)

    # Compute normalized input
    normalized_input0 = centered_input0 * inv_std

    # Compute output
    output1 = normalized_input0 * input2 + input3
    output2 = input1 + output1
    broadcast_output2 = tl.broadcast_to(output2, [RBLOCK])
    masked_broadcast_output2 = tl.where(r_mask, broadcast_output2, 0)
    sum_masked_broadcast_output2 = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast_output2, 0))
    mean_output2 = sum_masked_broadcast_output2 / 768.0
    centered_output2 = output2 - mean_output2
    squared_centered_output2 = centered_output2 * centered_output2
    broadcast_squared_centered_output2 = tl.broadcast_to(squared_centered_output2, [RBLOCK])
    masked_broadcast_squared_centered_output2 = tl.where(r_mask, broadcast_squared_centered_output2, 0)
    sum_masked_broadcast_squared_centered_output2 = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast_squared_centered_output2, 0))
    variance_output2 = sum_masked_broadcast_squared_centered_output2 / 768.0
    variance_output2_with_epsilon = variance_output2 + epsilon
    inv_std_output2 = tl.extra.cuda.libdevice.rsqrt(variance_output2_with_epsilon)
    normalized_output2 = (output2 - mean_output2) * inv_std_output2

    # Store results
    scale_factor = 0.0013020833333333333
    inv_std_scaled = inv_std * scale_factor
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_std, None)
    tl.store(in_out_ptr1 + (r1 + 768 * x0), normalized_output2, r_mask)
    tl.store(out_ptr3 + (x0), inv_std_scaled, None)
    tl.store(out_ptr0 + (x0), mean, None)