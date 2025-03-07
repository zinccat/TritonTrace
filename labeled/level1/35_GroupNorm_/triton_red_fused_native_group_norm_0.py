# From: 35_GroupNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_native_group_norm_0(input_ptr, output_mean_ptr, output_var_ptr, output_count_ptr, num_elements, num_groups, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    num_elements = 384
    num_groups = 174763
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    group_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_indices % 3
    x_group = (x_indices // 3)
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_linear_index = x_indices

    for group_offset in range(0, num_groups, RBLOCK):
        group_indices = group_offset + group_base
        group_mask = group_indices < num_groups
        group_linear_index = group_indices
        temp_index = group_linear_index + (num_groups * x_channel)
        max_index = tl.full([1, 1], 524288, tl.int32)
        index_mask = temp_index < max_index
        input_values = tl.load(input_ptr + ((524288 * x_group) + ((group_linear_index + (num_groups * x_channel)) % 524288)), group_mask & index_mask & x_mask, eviction_policy='evict_last', other=0.0)
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
        temp_mean = tl.where(group_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(group_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(group_mask & x_mask, temp_weight_next, temp_weight)

    final_mean, final_m2, final_weight = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    final_mean = final_mean[:, None]
    final_m2 = final_m2[:, None]
    final_weight = final_weight[:, None]
    tl.store(output_mean_ptr + (x_linear_index), final_mean, x_mask)
    tl.store(output_var_ptr + (x_linear_index), final_m2, x_mask)
    tl.store(output_count_ptr + (x_linear_index), final_weight, x_mask)