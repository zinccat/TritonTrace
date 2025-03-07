# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_layer_norm_1red_fused_native_layer_norm_1(
    input_output_ptr, input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 31360
    rnumel = 96
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = (x_index % 3136)
    x1 = x_index // 3136
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        input_data = tl.load(input_ptr0 + (x0 + 3136 * r2 + 301056 * x1), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_data = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_data, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

    mean_result, variance_result, weight_result = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    weight_result = weight_result[:, None]

    tl.store(output_ptr0 + (x3), mean_result, x_mask)
    epsilon = 1e-05
    normalized_variance = variance_result / 96.0 + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(normalized_variance)
    tl.debug_barrier()
    tl.store(input_output_ptr + (x3), inv_sqrt_variance, x_mask)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        input_data = tl.load(input_ptr0 + (x0 + 3136 * r2 + 301056 * x1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        scale_factor = tl.load(input_ptr1 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        bias_factor = tl.load(input_ptr2 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        centered_data = input_data - mean_result
        normalized_data = centered_data * inv_sqrt_variance
        scaled_data = normalized_data * scale_factor
        output_data = scaled_data + bias_factor
        tl.store(output_ptr1 + (r2 + 96 * x3), output_data, r_mask & x_mask)