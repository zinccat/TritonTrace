# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_52red_fused__native_batch_norm_legit_functional_52(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 10752
    reduction_elements = 123
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_indices // 672
    x_within_channel = x_indices % 672
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_full_indices = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_index = r_indices
        temp_index = r_index + 123 * x_channel
        max_index = tl.full([1, 1], 1960, tl.int32)
        index_mask = temp_index < max_index
        temp_data = tl.load(
            input_ptr + (x_within_channel + 672 * ((temp_index % 1960))),
            r_mask & index_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        temp_zero = tl.where(index_mask, zero_value, zero_broadcast)
        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        temp_one = tl.where(index_mask, one_value, one_broadcast)
        temp_data_broadcast = tl.broadcast_to(temp_data, [XBLOCK, RBLOCK])
        temp_zero_broadcast = tl.broadcast_to(temp_zero, [XBLOCK, RBLOCK])
        temp_one_broadcast = tl.broadcast_to(temp_one, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            temp_data_broadcast, temp_zero_broadcast, temp_one_broadcast
        )
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    temp_mean_final, temp_var_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    temp_mean_final_broadcast = temp_mean_final[:, None]
    temp_var_final_broadcast = temp_var_final[:, None]
    temp_weight_final_broadcast = temp_weight_final[:, None]
    tl.store(output_mean_ptr + (x_full_indices), temp_mean_final_broadcast, x_mask)
    tl.store(output_var_ptr + (x_full_indices), temp_var_final_broadcast, x_mask)
    tl.store(output_weight_ptr + (x_full_indices), temp_weight_final_broadcast, x_mask)