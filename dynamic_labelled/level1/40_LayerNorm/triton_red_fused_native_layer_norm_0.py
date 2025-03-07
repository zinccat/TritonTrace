# From: 40_LayerNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_layer_norm_0red_fused_native_layer_norm_0(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    x_num_elements, r_num_elements, X_BLOCK: tl.constexpr, R_BLOCK: tl.constexpr
):
    r_num_elements = 199729
    x_offset = tl.program_id(0) * X_BLOCK
    x_indices = x_offset + tl.arange(0, X_BLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    r_base_indices = tl.arange(0, R_BLOCK)[None, :]
    x_mod_21 = x_indices % 21
    x_div_21 = x_indices // 21
    temp_mean = tl.zeros([X_BLOCK, R_BLOCK], tl.float32)
    temp_m2 = tl.zeros([X_BLOCK, R_BLOCK], tl.float32)
    temp_weight = tl.zeros([X_BLOCK, R_BLOCK], tl.float32)
    x_indices_flat = x_indices

    for r_offset in range(0, r_num_elements, R_BLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < r_num_elements
        r_indices_flat = r_indices
        temp_index = r_indices_flat + 199729 * x_mod_21
        max_index = tl.full([1, 1], 4194304, tl.int32)
        index_mask = temp_index < max_index
        temp_load = tl.load(
            input_ptr + (4194304 * x_div_21 + ((r_indices_flat + 199729 * x_mod_21) % 4194304)),
            index_mask & r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        temp_zero = tl.where(index_mask, zero_value, zero_broadcast)
        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        temp_one = tl.where(index_mask, one_value, one_broadcast)
        temp_broadcast_load = tl.broadcast_to(temp_load, [X_BLOCK, R_BLOCK])
        temp_broadcast_zero = tl.broadcast_to(temp_zero, [X_BLOCK, R_BLOCK])
        temp_broadcast_one = tl.broadcast_to(temp_one, [X_BLOCK, R_BLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            temp_broadcast_load, temp_broadcast_zero, temp_broadcast_one
        )
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    temp_mean_final, temp_m2_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    temp_mean_final_broadcast = temp_mean_final[:, None]
    temp_m2_final_broadcast = temp_m2_final[:, None]
    temp_weight_final_broadcast = temp_weight_final[:, None]
    tl.store(output_mean_ptr + (x_indices_flat), temp_mean_final_broadcast, x_mask)
    tl.store(output_var_ptr + (x_indices_flat), temp_m2_final_broadcast, x_mask)
    tl.store(output_weight_ptr + (x_indices_flat), temp_weight_final_broadcast, x_mask)