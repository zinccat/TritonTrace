# From: 35_GroupNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_0(input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, kernel_size, total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = (x_indices % 3)
    x_sample = x_indices // 3
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_full_index = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_index = r_indices
        temp_index = r_index + x_channel * ((2 + 8 * kernel_size * kernel_size) // 3)
        max_index = 8 * kernel_size * kernel_size
        valid_mask = temp_index < max_index
        input_values = tl.load(input_ptr + (8 * x_sample * kernel_size * kernel_size + ((temp_index % (8 * kernel_size * kernel_size)))), r_mask & valid_mask & x_mask, eviction_policy='evict_last', other=0.0)
        zero_tensor = tl.full(input_values.shape, 0, input_values.dtype)
        valid_values = tl.where(valid_mask, 0.0, zero_tensor)
        valid_flags = tl.where(valid_mask, 1.0, zero_tensor)
        broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
        broadcasted_valid_values = tl.broadcast_to(valid_values, [XBLOCK, RBLOCK])
        broadcasted_valid_flags = tl.broadcast_to(valid_flags, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            broadcasted_values, broadcasted_valid_values, broadcasted_valid_flags
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
    tl.store(output_mean_ptr + (x_full_index), final_mean, x_mask)
    tl.store(output_var_ptr + (x_full_index), final_m2, x_mask)
    tl.store(output_weight_ptr + (x_full_index), final_weight, x_mask)