# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_layer_norm_43red_fused_native_layer_norm_43(
    input_output_ptr, input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 490
    rnumel = 1536
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        input_data = tl.load(input_ptr0 + (r1 + 1536 * x0), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_input, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

    mean_final, variance_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_final = mean_final[:, None]
    variance_final = variance_final[:, None]
    weight_final = weight_final[:, None]

    tl.store(output_ptr0 + (x0), mean_final, x_mask)
    scale_factor = 1536.0
    variance_adjusted = variance_final / scale_factor
    epsilon = 1e-05
    variance_adjusted += epsilon
    inverse_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)
    tl.debug_barrier()
    tl.store(input_output_ptr + (x0), inverse_sqrt_variance, x_mask)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        input_data = tl.load(input_ptr0 + (r1 + 1536 * x0), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        bias_data = tl.load(input_ptr1 + (r1), r_mask, eviction_policy='evict_last', other=0.0)
        gamma_data = tl.load(input_ptr2 + (r1), r_mask, eviction_policy='evict_last', other=0.0)
        centered_data = input_data - mean_final
        normalized_data = centered_data * inverse_sqrt_variance
        scaled_data = normalized_data * bias_data
        output_data = scaled_data + gamma_data
        tl.store(output_ptr1 + (r1 + 1536 * x0), output_data, r_mask & x_mask)