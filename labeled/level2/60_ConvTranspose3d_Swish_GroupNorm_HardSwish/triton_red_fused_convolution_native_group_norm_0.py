# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_convolution_native_group_norm_0(
    in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    rnumel = 492156
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x4 = x_index
    x0 = x_index % 4

    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r5 = r_index
        r3 = (r_index // 123039)

        input_value = tl.load(
            in_out_ptr0 + (r5 + (492156 * x4)), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        weight_value = tl.load(
            in_ptr0 + (r3 + (4 * x0)), 
            r_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        combined_value = input_value + weight_value
        swish_activation = tl.sigmoid(combined_value)
        swish_output = swish_activation * combined_value
        swish_broadcast = tl.broadcast_to(swish_output, [XBLOCK, RBLOCK])

        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            swish_broadcast, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )

        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

        tl.store(in_out_ptr0 + (r5 + (492156 * x4)), combined_value, r_mask & x_mask)

    mean_final, variance_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )

    mean_final = mean_final[:, None]
    variance_final = variance_final[:, None]

    tl.store(out_ptr0 + (x4), mean_final, x_mask)

    variance_scale = 492156.0
    variance_adjusted = variance_final / variance_scale
    epsilon = 1e-05
    variance_adjusted += epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)

    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), inv_stddev, x_mask)