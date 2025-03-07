# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_convolution_native_group_norm_0(
    in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 1024
    rnumel = 34848
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x4 = x_index
    x0 = x_index % 8

    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r5 = r_index
        r3 = (r_index // 4356)

        input_value = tl.load(in_out_ptr0 + (r5 + (34848 * x4)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        weight_value = tl.load(in_ptr0 + (r3 + (8 * x0)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        combined_value = input_value + weight_value

        half = 0.5
        scaled_value = combined_value * half
        sqrt_half = 0.7071067811865476
        erf_input = combined_value * sqrt_half
        erf_result = tl.extra.cuda.libdevice.erf(erf_input)
        one = 1.0
        erf_adjusted = erf_result + one
        gelu_output = scaled_value * erf_adjusted
        gelu_broadcast = tl.broadcast_to(gelu_output, [XBLOCK, RBLOCK])

        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            gelu_broadcast, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )

        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

        tl.store(in_out_ptr0 + (r5 + (34848 * x4)), combined_value, r_mask & x_mask)

    mean_final, variance_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )

    mean_final_broadcast = mean_final[:, None]
    variance_final_broadcast = variance_final[:, None]

    tl.store(out_ptr0 + (x4), mean_final_broadcast, x_mask)
    tl.store(out_ptr1 + (x4), variance_final_broadcast, x_mask)

    rnumel_float = 34848.0
    variance_normalized = variance_final_broadcast / rnumel_float
    epsilon = 1e-05
    variance_adjusted = variance_normalized + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)

    tl.store(out_ptr2 + (x4), inv_sqrt_variance, x_mask)