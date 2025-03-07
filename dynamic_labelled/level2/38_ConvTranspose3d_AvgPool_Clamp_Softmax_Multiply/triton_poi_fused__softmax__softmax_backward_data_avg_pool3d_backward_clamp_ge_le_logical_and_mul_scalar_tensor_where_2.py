# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_avg_pool3d_backward_clamp_ge_le_logical_and_mul_scalar_tensor_where_2(
    input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, stride_z, stride_y, stride_x, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    index_z = block_indices % kernel_size_z
    index_y = (block_indices // kernel_size_z) % kernel_size_y
    index_x = (block_indices // kernel_size_y) % kernel_size_x
    batch_index = block_indices // kernel_size_z
    linear_index = block_indices

    input_value = tl.load(
        input_ptr + (
            kernel_size_y * (
                ((0) * ((0) >= (index_y // 2)) + (index_y // 2) * ((index_y // 2) > (0)))
                * (((0) * ((0) >= (index_y // 2)) + (index_y // 2) * ((index_y // 2) > (0))) <= ((-1) + ((kernel_size_y) * ((kernel_size_y) <= (1 + (index_y // 2))) + (1 + (index_y // 2)) * ((1 + (index_y // 2)) < (kernel_size_y)))))
                + ((-1) + ((kernel_size_y) * ((kernel_size_y) <= (1 + (index_y // 2))) + (1 + (index_y // 2)) * ((1 + (index_y // 2)) < (kernel_size_y))))
                * (((-1) + ((kernel_size_y) * ((kernel_size_y) <= (1 + (index_y // 2))) + (1 + (index_y // 2)) * ((1 + (index_y // 2)) < (kernel_size_y)))) < (((0) * ((0) >= (index_y // 2)) + (index_y // 2) * ((index_y // 2) > (0)))))
            )
            + kernel_size_y * kernel_size_z * (
                ((0) * ((0) >= (index_x // 2)) + (index_x // 2) * ((index_x // 2) > (0)))
                * (((0) * ((0) >= (index_x // 2)) + (index_x // 2) * ((index_x // 2) > (0))) <= ((-1) + ((kernel_size_x) * ((kernel_size_x) <= (1 + (index_x // 2))) + (1 + (index_x // 2)) * ((1 + (index_x // 2)) < (kernel_size_x)))))
                + ((-1) + ((kernel_size_x) * ((kernel_size_x) <= (1 + (index_x // 2))) + (1 + (index_x // 2)) * ((1 + (index_x // 2)) < (kernel_size_x))))
                * (((-1) + ((kernel_size_x) * ((kernel_size_x) <= (1 + (index_x // 2))) + (1 + (index_x // 2)) * ((1 + (index_x // 2)) < (kernel_size_x)))) < (((0) * ((0) >= (index_x // 2)) + (index_x // 2) * ((index_x // 2) > (0)))))
            )
            + kernel_size_x * batch_index * kernel_size_y * kernel_size_z
            + (
                ((0) * ((0) >= (index_z // 2)) + (index_z // 2) * ((index_z // 2) > (0)))
                * (((0) * ((0) >= (index_z // 2)) + (index_z // 2) * ((index_z // 2) > (0))) <= ((-1) + ((kernel_size_z) * ((kernel_size_z) <= (1 + (index_z // 2))) + (1 + (index_z // 2)) * ((1 + (index_z // 2)) < (kernel_size_z)))))
                + ((-1) + ((kernel_size_z) * ((kernel_size_z) <= (1 + (index_z // 2))) + (1 + (index_z // 2)) * ((1 + (index_z // 2)) < (kernel_size_z))))
                * (((-1) + ((kernel_size_z) * ((kernel_size_z) <= (1 + (index_z // 2))) + (1 + (index_z // 2)) * ((1 + (index_z // 2)) < (kernel_size_z)))) < (((0) * ((0) >= (index_z // 2)) + (index_z // 2) * ((index_z // 2) > (0)))))
            )
        ),
        valid_mask,
        eviction_policy='evict_last'
    )

    avg_pool_value = input_value / 8

    y_condition = ((0) * ((0) >= (index_y // 2)) + (index_y // 2) * ((index_y // 2) > (0)))
    y_limit = ((kernel_size_y) * ((kernel_size_y) <= (1 + (index_y // 2))) + (1 + (index_y // 2)) * ((1 + (index_y // 2)) < (kernel_size_y)))
    y_valid = y_condition < y_limit

    x_condition = ((0) * ((0) >= (index_x // 2)) + (index_x // 2) * ((index_x // 2) > (0)))
    x_limit = ((kernel_size_x) * ((kernel_size_x) <= (1 + (index_x // 2))) + (1 + (index_x // 2)) * ((1 + (index_x // 2)) < (kernel_size_x)))
    x_valid = x_condition < x_limit

    z_condition = ((0) * ((0) >= (index_z // 2)) + (index_z // 2) * ((index_z // 2) > (0)))
    z_limit = ((kernel_size_z) * ((kernel_size_z) <= (1 + (index_z // 2))) + (1 + (index_z // 2)) * ((1 + (index_z // 2)) < (kernel_size_z)))
    z_valid = z_condition < z_limit

    valid_yx = y_valid & x_valid
    valid_xyz = valid_yx & z_valid

    zero_value = 0.0
    result_value = tl.where(valid_xyz, avg_pool_value, zero_value)

    tl.store(output_ptr + (linear_index), result_value, valid_mask)