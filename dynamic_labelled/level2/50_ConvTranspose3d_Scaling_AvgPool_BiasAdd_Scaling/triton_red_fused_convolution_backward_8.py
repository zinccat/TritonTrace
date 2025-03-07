# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_8(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4, kernel_size_5,
    x_num_elements, r_num_elements, X_BLOCK: tl.constexpr, R_BLOCK: tl.constexpr
):
    x_num_elements = 1968
    x_offset = tl.program_id(0) * X_BLOCK
    x_indices = x_offset + tl.arange(0, X_BLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    r_base = tl.arange(0, R_BLOCK)[None, :]
    x_div_16 = x_indices // 16
    x_mod_16 = x_indices % 16
    temp_accumulator = tl.full([X_BLOCK, R_BLOCK], 0, tl.float32)
    x_indices_flat = x_indices

    for r_offset in range(0, r_num_elements, R_BLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < r_num_elements
        r_indices_flat = r_indices

        temp_index = r_indices_flat + x_div_16 * (
            triton_helpers.div_floor_integer(
                122 + (-1 * kernel_size_0) + (-4 * kernel_size_0 * kernel_size_2 * kernel_size_2) +
                2 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_2 +
                (-8 * kernel_size_0 * kernel_size_1 * kernel_size_2) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2,
                123
            )
        )

        temp_condition = temp_index < (
            (-1 * kernel_size_0) + (-4 * kernel_size_0 * kernel_size_2 * kernel_size_2) +
            2 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_2 +
            (-8 * kernel_size_0 * kernel_size_1 * kernel_size_2) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2
        )

        temp_load = tl.load(
            input_ptr + (
                (-1 * x_mod_16) + (-1 * (((temp_index // kernel_size_3) % kernel_size_3))) +
                (-16 * (((temp_index // ((-1) + (-4 * kernel_size_2 * kernel_size_2) + 2 * kernel_size_1 + 4 * kernel_size_2 + (-8 * kernel_size_1 * kernel_size_2) + 8 * kernel_size_1 * kernel_size_2 * kernel_size_2)) % kernel_size_0)))) +
                (-64 * kernel_size_2 * kernel_size_2 * (((temp_index // ((-1) + (-4 * kernel_size_2 * kernel_size_2) + 2 * kernel_size_1 + 4 * kernel_size_2 + (-8 * kernel_size_1 * kernel_size_2) + 8 * kernel_size_1 * kernel_size_2 * kernel_size_2)) % kernel_size_0))) +
                (-4 * kernel_size_2 * (((temp_index // kernel_size_5) % kernel_size_4))) +
                (-4 * x_mod_16 * kernel_size_2 * kernel_size_2) + 2 * kernel_size_1 * x_mod_16 +
                2 * kernel_size_2 * (((temp_index // kernel_size_3) % kernel_size_3)) + 4 * kernel_size_2 * x_mod_16 +
                4 * kernel_size_2 * kernel_size_2 * (((temp_index // kernel_size_5) % kernel_size_4)) +
                32 * kernel_size_1 * (((temp_index // ((-1) + (-4 * kernel_size_2 * kernel_size_2) + 2 * kernel_size_1 + 4 * kernel_size_2 + (-8 * kernel_size_1 * kernel_size_2) + 8 * kernel_size_1 * kernel_size_2 * kernel_size_2)) % kernel_size_0)) +
                64 * kernel_size_2 * (((temp_index // ((-1) + (-4 * kernel_size_2 * kernel_size_2) + 2 * kernel_size_1 + 4 * kernel_size_2 + (-8 * kernel_size_1 * kernel_size_2) + 8 * kernel_size_1 * kernel_size_2 * kernel_size_2)) % kernel_size_0)) +
                (-128 * kernel_size_1 * kernel_size_2 * (((temp_index // ((-1) + (-4 * kernel_size_2 * kernel_size_2) + 2 * kernel_size_1 + 4 * kernel_size_2 + (-8 * kernel_size_1 * kernel_size_2) + 8 * kernel_size_1 * kernel_size_2 * kernel_size_2)) % kernel_size_0))) +
                (-8 * kernel_size_1 * kernel_size_2 * x_mod_16) + 8 * kernel_size_1 * x_mod_16 * kernel_size_2 * kernel_size_2 +
                128 * kernel_size_1 * kernel_size_2 * kernel_size_2 * (((temp_index // ((-1) + (-4 * kernel_size_2 * kernel_size_2) + 2 * kernel_size_1 + 4 * kernel_size_2 + (-8 * kernel_size_1 * kernel_size_2) + 8 * kernel_size_1 * kernel_size_2 * kernel_size_2)) % kernel_size_0)) +
                (((temp_index % kernel_size_3))) + (((temp_index // kernel_size_5) % kernel_size_4))
            ),
            r_mask & temp_condition & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_condition, [X_BLOCK, R_BLOCK])
        temp_accumulator_update = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(r_mask & x_mask, temp_accumulator_update, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (x_indices_flat), temp_sum, x_mask)