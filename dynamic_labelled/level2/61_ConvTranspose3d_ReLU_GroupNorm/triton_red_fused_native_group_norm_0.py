# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_0red_fused_native_group_norm_0(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, kernel_size_z, kernel_size_y, num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    x_channel = (x_indices % 4)
    x_batch = x_indices // 4
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < num_elements_r
        r_flat_index = r_indices

        input_index = (
            2 * (
                (((r_flat_index % (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) // (2 + kernel_size_z)) % (2 + kernel_size_z))
                + 4 * (
                    (((r_flat_index % (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) // (4 + kernel_size_z * kernel_size_z + 4 * kernel_size_z)) % (2 + kernel_size_y))
                )
                + 8 * (
                    (((r_flat_index + 32 * x_channel + 8 * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_y * x_channel + 32 * kernel_size_z * x_channel + 4 * kernel_size_y * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_z * kernel_size_y * x_channel) // (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) % 16))
                + 128 * x_batch
                + kernel_size_z * (
                    (((r_flat_index % (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) // (2 + kernel_size_z)) % (2 + kernel_size_z))
                )
                + kernel_size_z * kernel_size_z * (
                    (((r_flat_index % (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) // (4 + kernel_size_z * kernel_size_z + 4 * kernel_size_z)) % (2 + kernel_size_y))
                )
                + 2 * kernel_size_z * kernel_size_z * (
                    (((r_flat_index + 32 * x_channel + 8 * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_y * x_channel + 32 * kernel_size_z * x_channel + 4 * kernel_size_y * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_z * kernel_size_y * x_channel) // (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) % 16))
                + 4 * kernel_size_z * (
                    (((r_flat_index % (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) // (4 + kernel_size_z * kernel_size_z + 4 * kernel_size_z)) % (2 + kernel_size_y))
                )
                + 4 * kernel_size_y * (
                    (((r_flat_index + 32 * x_channel + 8 * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_y * x_channel + 32 * kernel_size_z * x_channel + 4 * kernel_size_y * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_z * kernel_size_y * x_channel) // (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) % 16))
                + 8 * kernel_size_z * (
                    (((r_flat_index + 32 * x_channel + 8 * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_y * x_channel + 32 * kernel_size_z * x_channel + 4 * kernel_size_y * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_z * kernel_size_y * x_channel) // (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) % 16))
                + 32 * x_batch * kernel_size_z * kernel_size_z
                + 64 * kernel_size_y * x_batch
                + 128 * kernel_size_z * x_batch
                + kernel_size_y * kernel_size_z * kernel_size_z * (
                    (((r_flat_index + 32 * x_channel + 8 * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_y * x_channel + 32 * kernel_size_z * x_channel + 4 * kernel_size_y * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_z * kernel_size_y * x_channel) // (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) % 16))
                + 4 * kernel_size_z * kernel_size_y * (
                    (((r_flat_index + 32 * x_channel + 8 * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_y * x_channel + 32 * kernel_size_z * x_channel + 4 * kernel_size_y * x_channel * kernel_size_z * kernel_size_z + 16 * kernel_size_z * kernel_size_y * x_channel) // (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) % 16))
                + 16 * kernel_size_y * x_batch * kernel_size_z * kernel_size_z
                + 64 * kernel_size_z * kernel_size_y * x_batch
                + (((r_flat_index % (8 + 2 * kernel_size_z * kernel_size_z + 4 * kernel_size_y + 8 * kernel_size_z + kernel_size_y * kernel_size_z * kernel_size_z + 4 * kernel_size_z * kernel_size_y)) % (2 + kernel_size_z)))
        )

        input_data = tl.load(input_ptr + input_index, rmask & x_mask, eviction_policy='evict_last', other=0.0)
        max_value = tl.full([1, 1], 0, tl.int32)
        max_data = triton_helpers.maximum(max_value, input_data)
        broadcast_max = tl.broadcast_to(max_data, [XBLOCK, RBLOCK])

        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_reduce(
            broadcast_max, temp_mean, temp_m2, temp_weight, r_offset == 0
        )

        temp_mean = tl.where(rmask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(rmask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(rmask & x_mask, temp_weight_next, temp_weight)

    mean, variance, weight = triton_helpers.welford(temp_mean, temp_m2, temp_weight, 1)
    mean = mean[:, None]
    variance = variance[:, None]
    weight = weight[:, None]

    tl.store(output_mean_ptr + x_flat_index, mean, x_mask)
    tl.store(output_var_ptr + x_flat_index, variance, x_mask)
    tl.store(output_weight_ptr + x_flat_index, weight, x_mask)