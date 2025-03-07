# From: 33_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_0(
    input_ptr, output_mean_ptr, output_variance_ptr, output_weight_ptr, kernel_size_0, kernel_size_1, num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_x = 384
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_index // 64
    x_within_channel = x_index % 64
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_index

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < num_elements_r
        r_flat_index = r_index
        temp_index = r_flat_index + x_channel * ((5 + kernel_size_0 * kernel_size_1 * kernel_size_1) // 6)
        total_elements = kernel_size_0 * kernel_size_1 * kernel_size_1
        valid_index = temp_index < total_elements
        input_value = tl.load(
            input_ptr + (
                x_within_channel * kernel_size_1 * kernel_size_1 +
                64 * kernel_size_1 * kernel_size_1 * (((temp_index // (kernel_size_1 * kernel_size_1)) % kernel_size_0)) +
                (temp_index % (kernel_size_1 * kernel_size_1))
            ),
            valid_index & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        zero_condition = tl.where(valid_index, zero_value, zero_broadcast)
        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        one_condition = tl.where(valid_index, one_value, one_broadcast)
        input_broadcast = tl.broadcast_to(input_value, [XBLOCK, RBLOCK])
        zero_condition_broadcast = tl.broadcast_to(zero_condition, [XBLOCK, RBLOCK])
        one_condition_broadcast = tl.broadcast_to(one_condition, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            input_broadcast, zero_condition_broadcast, one_condition_broadcast
        )
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    final_mean, final_m2, final_weight = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    final_mean_broadcast = final_mean[:, None]
    final_m2_broadcast = final_m2[:, None]
    final_weight_broadcast = final_weight[:, None]
    tl.store(output_mean_ptr + (x_flat_index), final_mean_broadcast, x_mask)
    tl.store(output_variance_ptr + (x_flat_index), final_m2_broadcast, x_mask)
    tl.store(output_weight_ptr + (x_flat_index), final_weight_broadcast, x_mask)