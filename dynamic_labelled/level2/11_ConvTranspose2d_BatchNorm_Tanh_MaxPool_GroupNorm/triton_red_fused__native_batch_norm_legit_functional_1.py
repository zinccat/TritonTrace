# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, kernel_size, total_elements, reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 384
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_indices // 64
    x_pixel = x_indices % 64
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_flat_index = r_indices
        temp_index = r_flat_index + x_channel * ((5 + 4096 * kernel_size) // 6)
        max_index = 4096 * kernel_size
        valid_mask = temp_index < max_index
        input_value = tl.load(
            input_ptr + (4096 * x_pixel + 262144 * (((temp_index // 4096) % kernel_size)) + (temp_index % 4096)),
            valid_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        valid_zero = tl.where(valid_mask, zero_value, zero_broadcast)
        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        valid_one = tl.where(valid_mask, one_value, one_broadcast)
        input_broadcast = tl.broadcast_to(input_value, [XBLOCK, RBLOCK])
        valid_zero_broadcast = tl.broadcast_to(valid_zero, [XBLOCK, RBLOCK])
        valid_one_broadcast = tl.broadcast_to(valid_one, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            input_broadcast, valid_zero_broadcast, valid_one_broadcast
        )
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    temp_mean_final, temp_var_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    temp_mean_final = temp_mean_final[:, None]
    temp_var_final = temp_var_final[:, None]
    temp_weight_final = temp_weight_final[:, None]
    tl.store(output_mean_ptr + (x_flat_index), temp_mean_final, x_mask)
    tl.store(output_var_ptr + (x_flat_index), temp_var_final, x_mask)
    tl.store(output_weight_ptr + (x_flat_index), temp_weight_final, x_mask)