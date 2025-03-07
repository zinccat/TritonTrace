# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_43(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 7840
    rnumel = 192
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_col = (x_index % 784)
    x_row = x_index // 784
    tmp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_flat_index = r_index

        tmp_input0 = tl.load(
            in_ptr0 + (r_flat_index + 192 * (((((25 + (x_col % 28)) % 28)) % 7)) + 1344 * (((((25 + (x_col // 28)) % 28)) % 7)) + 9408 * ((((25 + (x_col % 28)) % 28)) // 7) + 37632 * (triton_helpers.div_floor_integer(((25 + (x_col // 28)) % 28), 7)) + 150528 * x_row),
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
            in_ptr0 + (r_flat_index + 192 * (((((25 + (x_col % 28)) % 28)) % 7)) + 1344 * (((((25 + (x_col // 28)) % 28)) % 7)) + 9408 * ((((25 + (x_col % 28)) % 28)) // 7) + 37632 * (triton_helpers.div_floor_integer(((25 + (x_col // 28)) % 28), 7)) + 150528 * x_row),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        tmp_input1 = tl.load(in_ptr1 + (r_flat_index), r_mask, eviction_policy='evict_last', other=0.0)
        tmp_grad_input = tl.load(in_out_ptr0 + (r_flat_index + 192 * x_flat_index), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        tmp_grad_weight = tl.load(in_ptr2 + (r_flat_index), r_mask, eviction_policy='evict_last', other=0.0)
        tmp_grad_bias = tl.load(in_ptr3 + (r_flat_index), r_mask, eviction_policy='evict_last', other=0.0)

        tmp_sum = tmp_input0 + tmp_input1
        tmp_centered = tmp_sum - tmp_mean_final
        tmp_var = 192.0
        tmp_mean_var = tmp_m2_final / tmp_var
        tmp_epsilon = 1e-05
        tmp_var_eps = tmp_mean_var + tmp_epsilon
        tmp_inv_std = tl.extra.cuda.libdevice.rsqrt(tmp_var_eps)
        tmp_normalized = tmp_centered * tmp_inv_std
        tmp_scaled_grad = tmp_normalized * tmp_grad_weight
        tmp_grad_input_update = tmp_grad_input + tmp_scaled_grad + tmp_grad_bias
        tl.store(out_ptr2 + (r_flat_index + 192 * x_flat_index), tmp_normalized, r_mask & x_mask)
        tl.store(in_out_ptr0 + (r_flat_index + 192 * x_flat_index), tmp_grad_input_update, r_mask & x_mask)

    tmp_var = 192.0
    tmp_mean_var = tmp_m2_final / tmp_var
    tmp_epsilon = 1e-05
    tmp_var_eps = tmp_mean_var + tmp_epsilon
    tmp_inv_std = tl.extra.cuda.libdevice.rsqrt(tmp_var_eps)
    tmp_scale_factor = 0.005208333333333333
    tmp_scaled_inv_std = tmp_inv_std * tmp_scale_factor
    tl.store(out_ptr3 + (x_flat_index), tmp_scaled_inv_std, x_mask)