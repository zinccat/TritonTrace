# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_9red_fused__native_batch_norm_legit_functional_9(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 41984
    reduction_elements = 612
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_64 = x_indices // 64
    x_mod_64 = x_indices % 64
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_full_indices = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_full_indices = r_indices
        temp_index = r_full_indices + 612 * x_div_64
        max_index = tl.full([1, 1], 401408, tl.int32)
        index_mask = temp_index < max_index
        loaded_values = tl.load(
            input_ptr + (x_mod_64 + 64 * ((temp_index % 401408))),
            r_mask & index_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        zero_tensor = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        ones_tensor = tl.full(loaded_values.shape, 1, loaded_values.dtype)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcasted_zeros = tl.broadcast_to(zero_tensor, [XBLOCK, RBLOCK])
        broadcasted_ones = tl.broadcast_to(ones_tensor, [XBLOCK, RBLOCK])

        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            broadcasted_values, broadcasted_zeros, broadcasted_ones
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