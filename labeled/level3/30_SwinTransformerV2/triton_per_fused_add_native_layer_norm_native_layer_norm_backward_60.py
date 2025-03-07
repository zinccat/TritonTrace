# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_60(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r2 = r_index
    x0 = (x_index % 196)
    x1 = x_index // 196
    x3 = x_index

    # Load input data with masking
    input0 = tl.load(
        in_ptr0 + (r2 + 384 * (((x0 % 14)) % 7) + 2688 * (((x0 // 14) % 7)) + 18816 * (((x0 % 14)) // 7) + 37632 * (x0 // 98) + 75264 * x1),
        r_mask,
        other=0.0
    )
    input1 = tl.load(in_ptr1 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
    in_out_data = tl.load(in_out_ptr0 + (r2 + 384 * x3), r_mask, other=0.0)
    input2 = tl.load(in_ptr2 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
    input3 = tl.load(in_ptr3 + (r2), r_mask, eviction_policy='evict_last', other=0.0)

    # Compute intermediate values
    sum_inputs = input0 + input1
    broadcast_sum = tl.broadcast_to(sum_inputs, [RBLOCK])
    masked_broadcast = tl.where(r_mask, broadcast_sum, 0)
    broadcast_masked = tl.broadcast_to(masked_broadcast, [RBLOCK])
    masked_broadcast2 = tl.where(r_mask, broadcast_masked, 0)
    sum_masked = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast2, 0))
    rblock_size = tl.full([1], 384, tl.int32)
    rblock_size_float = rblock_size.to(tl.float32)
    mean = sum_masked / rblock_size_float
    centered = masked_broadcast - mean
    squared_centered = centered * centered
    broadcast_squared = tl.broadcast_to(squared_centered, [RBLOCK])
    masked_squared = tl.where(r_mask, broadcast_squared, 0)
    sum_squared = triton_helpers.promote_to_tensor(tl.sum(masked_squared, 0))
    variance = (sum_inputs - mean) / rblock_size_float
    epsilon = 1e-05
    variance_epsilon = sum_squared / 384.0 + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance_epsilon)
    normalized = (variance - mean) * inv_std
    scaled = normalized * input2
    output = scaled + input3
    updated_in_out = in_out_data + output
    scale_factor = 0.0026041666666666665
    gamma = inv_std * scale_factor

    # Store results
    tl.store(out_ptr2 + (r2 + 384 * x3), normalized, r_mask)
    tl.store(in_out_ptr0 + (r2 + 384 * x3), updated_in_out, r_mask)
    tl.store(out_ptr3 + (x3), gamma, None)