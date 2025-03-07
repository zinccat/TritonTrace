# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_20(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 31360
    rnumel = 96
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_col = (x_index % 3136)
    x_row = x_index // 3136
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        tmp0 = tl.load(
            in_ptr0 + (r2 + 96 * (((((53 + ((x_col % 56))) % 56)) % 7)) + 672 * (((((53 + (x_col // 56)) % 56)) % 7)) + 4704 * ((((53 + ((x_col % 56))) % 56)) // 7) + 37632 * (triton_helpers.div_floor_integer(((53 + (x_col // 56)) % 56), 7)) + 301056 * x_row),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        tmp1 = tl.load(in_ptr1 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            tmp3, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

    mean_final, variance_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_broadcast = mean_final[:, None]
    variance_broadcast = variance_final[:, None]

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        tmp7 = tl.load(
            in_ptr0 + (r2 + 96 * (((((53 + ((x_col % 56))) % 56)) % 7)) + 672 * (((((53 + (x_col // 56)) % 56)) % 7)) + 4704 * ((((53 + ((x_col % 56))) % 56)) // 7) + 37632 * (triton_helpers.div_floor_integer(((53 + (x_col // 56)) % 56), 7)) + 301056 * x_row),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        tmp8 = tl.load(in_ptr1 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_out_ptr0 + (r2 + 96 * x3), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr3 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - mean_broadcast
        epsilon = 1e-05
        normalized_value = tmp10 * tl.extra.cuda.libdevice.rsqrt(tmp5 / 96.0 + epsilon)
        scaled_gradient = normalized_value * tmp18
        gradient_update = scaled_gradient + tmp20
        updated_value = tmp17 + gradient_update
        tl.store(out_ptr2 + (r2 + 96 * x3), normalized_value, r_mask & x_mask)
        tl.store(in_out_ptr0 + (r2 + 96 * x3), updated_value, r_mask & x_mask)

    scale_factor = tl.extra.cuda.libdevice.rsqrt(tmp5 / 96.0 + 1e-05) * 0.010416666666666666
    tl.store(out_ptr3 + (x3), scale_factor, x_mask)