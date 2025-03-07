# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_66(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 1960
    rnumel = 384
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_col = (x_index % 196)
    x_row = x_index // 196
    tmp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_flat_index = r_index
        tmp_input0 = tl.load(
            in_ptr0 + (r_flat_index + 384 * (((((11 + (x_col % 14)) % 14)) % 7)) + 2688 * (((((11 + (x_col // 14)) % 14)) % 7)) + 18816 * ((((11 + (x_col % 14)) % 14)) // 7) + 37632 * (triton_helpers.div_floor_integer(((11 + (x_col // 14)) % 14), 7)) + 75264 * x_row),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        tmp_input1 = tl.load(in_ptr1 + (r_flat_index), r_mask, eviction_policy='evict_last', other=0.0)
        tmp_sum = tmp_input0 + tmp_input1
        tmp_broadcast = tl.broadcast_to(tmp_sum, [XBLOCK, RBLOCK])
        tmp_mean_next, tmp_m2_next, tmp_weight_next = triton_helpers.welford_reduce(
            tmp_broadcast, tmp_mean, tmp_m2, tmp_weight, r_offset == 0
        )
        tmp_mean = tl.where(r_mask & x_mask, tmp_mean_next, tmp_mean)
        tmp_m2 = tl.where(r_mask & x_mask, tmp_m2_next, tmp_m2)
        tmp_weight = tl.where(r_mask & x_mask, tmp_weight_next, tmp_weight)

    tmp_mean_final, tmp_m2_final, tmp_weight_final = triton_helpers.welford(
        tmp_mean, tmp_m2, tmp_weight, 1
    )
    tmp_mean_final = tmp_mean_final[:, None]
    tmp_m2_final = tmp_m2_final[:, None]

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_flat_index = r_index
        tmp_input0 = tl.load(
            in_ptr0 + (r_flat_index + 384 * (((((11 + (x_col % 14)) % 14)) % 7)) + 2688 * (((((11 + (x_col // 14)) % 14)) % 7)) + 18816 * ((((11 + (x_col % 14)) % 14)) // 7) + 37632 * (triton_helpers.div_floor_integer(((11 + (x_col // 14)) % 14), 7)) + 75264 * x_row),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        tmp_input1 = tl.load(in_ptr1 + (r_flat_index), r_mask, eviction_policy='evict_last', other=0.0)
        tmp_input2 = tl.load(in_out_ptr0 + (r_flat_index + 384 * x_flat_index), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        tmp_input3 = tl.load(in_ptr2 + (r_flat_index), r_mask, eviction_policy='evict_last', other=0.0)
        tmp_input4 = tl.load(in_ptr3 + (r_flat_index), r_mask, eviction_policy='evict_last', other=0.0)
        tmp_sum = tmp_input0 + tmp_input1
        tmp_centered = tmp_sum - tmp_mean_final
        tmp_denom = 384.0
        tmp_var = tmp_m2_final / tmp_denom
        epsilon = 1e-05
        tmp_var_eps = tmp_var + epsilon
        tmp_inv_std = tl.extra.cuda.libdevice.rsqrt(tmp_var_eps)
        tmp_normalized = tmp_centered * tmp_inv_std
        tmp_scaled = tmp_normalized * tmp_input3
        tmp_output = tmp_scaled + tmp_input4
        tmp_updated_input2 = tmp_input2 + tmp_output
        tl.store(out_ptr2 + (r_flat_index + 384 * x_flat_index), tmp_normalized, r_mask & x_mask)
        tl.store(in_out_ptr0 + (r_flat_index + 384 * x_flat_index), tmp_updated_input2, r_mask & x_mask)

    denom = 384.0
    var = tmp_m2_final / denom
    epsilon = 1e-05
    var_eps = var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(var_eps)
    scale_factor = 0.0026041666666666665
    final_output = inv_std * scale_factor
    tl.store(out_ptr3 + (x_flat_index), final_output, x_mask)