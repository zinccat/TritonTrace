# From: 40_LayerNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_native_layer_norm_0(input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, num_elements, num_reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    num_elements = 336
    num_reduction_elements = 199729
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_col = x_indices % 21
    x_row = (x_indices // 21)
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for r_offset in range(0, num_reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_reduction_elements
        r_flat_index = r_indices
        temp_index = r_flat_index + (num_reduction_elements * x_col)
        max_index = tl.full([1, 1], 4194304, tl.int32)
        index_mask = temp_index < max_index
        input_values = tl.load(input_ptr + ((4194304 * x_row) + ((r_flat_index + (num_reduction_elements * x_col)) % 4194304)), r_mask & index_mask & x_mask, eviction_policy='evict_last', other=0.0)
        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        zero_masked = tl.where(index_mask, zero_value, zero_broadcast)
        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        one_masked = tl.where(index_mask, one_value, one_broadcast)
        input_broadcast = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
        zero_masked_broadcast = tl.broadcast_to(zero_masked, [XBLOCK, RBLOCK])
        one_masked_broadcast = tl.broadcast_to(one_masked, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            input_broadcast, zero_masked_broadcast, one_masked_broadcast
        )
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    final_mean, final_m2, final_weight = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    final_mean = final_mean[:, None]
    final_m2 = final_m2[:, None]
    final_weight = final_weight[:, None]
    tl.store(output_mean_ptr + (x_flat_index), final_mean, x_mask)
    tl.store(output_var_ptr + (x_flat_index), final_m2, x_mask)
    tl.store(output_weight_ptr + (x_flat_index), final_weight, x_mask)