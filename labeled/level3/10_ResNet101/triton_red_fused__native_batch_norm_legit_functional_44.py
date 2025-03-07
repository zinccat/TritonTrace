# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_44red_fused__native_batch_norm_legit_functional_44(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_x = 2048
    num_elements_r = 123
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_indices // 512
    x_within_channel = x_indices % 512
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_full_indices = x_indices

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_full_indices = r_indices
        temp_index = r_full_indices + 123 * x_channel
        max_index = tl.full([1, 1], 490, tl.int32)
        index_mask = temp_index < max_index
        temp_values = tl.load(
            input_ptr + (x_within_channel + 512 * ((temp_index % 490))),
            r_mask & index_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        zero_tensor = tl.full(temp_values.shape, 0, temp_values.dtype)
        one_tensor = tl.full(temp_values.shape, 1, temp_values.dtype)
        temp_zeros = tl.where(index_mask, 0.0, zero_tensor)
        temp_ones = tl.where(index_mask, 1.0, one_tensor)
        broadcast_values = tl.broadcast_to(temp_values, [XBLOCK, RBLOCK])
        broadcast_zeros = tl.broadcast_to(temp_zeros, [XBLOCK, RBLOCK])
        broadcast_ones = tl.broadcast_to(temp_ones, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            broadcast_values, broadcast_zeros, broadcast_ones
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
    tl.store(output_mean_ptr + (x_full_indices), final_mean, x_mask)
    tl.store(output_var_ptr + (x_full_indices), final_m2, x_mask)
    tl.store(output_weight_ptr + (x_full_indices), final_weight, x_mask)