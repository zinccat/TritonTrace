# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_2red_fused__native_batch_norm_legit_functional_2(
    input_ptr, output_mean_ptr, output_variance_ptr, output_count_ptr, kernel_size_dim0, kernel_size_dim1, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 3984
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_dim0 = (input_indices % 249)
    input_dim1 = input_indices // 249
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_linear_index = input_indices

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_dim0 = reduction_indices
        temp_index = reduction_dim0 + input_dim0 * (
            triton_helpers.div_floor_integer(
                248 + ((-1) * kernel_size_dim0) + ((-12) * kernel_size_dim0 * kernel_size_dim1 * kernel_size_dim1) + 
                6 * kernel_size_dim0 * kernel_size_dim1 + 8 * kernel_size_dim0 * kernel_size_dim1 * kernel_size_dim1 * kernel_size_dim1, 
                249
            )
        )
        temp_limit = ((-1) * kernel_size_dim0) + ((-12) * kernel_size_dim0 * kernel_size_dim1 * kernel_size_dim1) + 
                     6 * kernel_size_dim0 * kernel_size_dim1 + 8 * kernel_size_dim0 * kernel_size_dim1 * kernel_size_dim1 * kernel_size_dim1
        temp_mask = temp_index < temp_limit

        temp_load = tl.load(
            input_ptr + (
                ((-1) * input_dim1) + 
                ((-1) * (((temp_index // ((-1) + 2 * kernel_size_dim1)) % ((-1) + 2 * kernel_size_dim1)))) + 
                ((-16) * (((temp_index // ((-1) + ((-12) * kernel_size_dim1 * kernel_size_dim1) + 6 * kernel_size_dim1 + 8 * kernel_size_dim1 * kernel_size_dim1 * kernel_size_dim1)) % kernel_size_dim0))) + 
                ((-192) * kernel_size_dim1 * kernel_size_dim1 * (((temp_index // ((-1) + ((-12) * kernel_size_dim1 * kernel_size_dim1) + 6 * kernel_size_dim1 + 8 * kernel_size_dim1 * kernel_size_dim1 * kernel_size_dim1)) % kernel_size_dim0))) + 
                ((-12) * input_dim1 * kernel_size_dim1 * kernel_size_dim1) + 
                ((-4) * kernel_size_dim1 * (((temp_index // (1 + ((-4) * kernel_size_dim1) + 4 * kernel_size_dim1 * kernel_size_dim1)) % ((-1) + 2 * kernel_size_dim1)))) + 
                2 * kernel_size_dim1 * (((temp_index // ((-1) + 2 * kernel_size_dim1)) % ((-1) + 2 * kernel_size_dim1))) + 
                4 * kernel_size_dim1 * kernel_size_dim1 * (((temp_index // (1 + ((-4) * kernel_size_dim1) + 4 * kernel_size_dim1 * kernel_size_dim1)) % ((-1) + 2 * kernel_size_dim1))) + 
                6 * kernel_size_dim1 * input_dim1 + 
                8 * input_dim1 * kernel_size_dim1 * kernel_size_dim1 * kernel_size_dim1 + 
                96 * kernel_size_dim1 * (((temp_index // ((-1) + ((-12) * kernel_size_dim1 * kernel_size_dim1) + 6 * kernel_size_dim1 + 8 * kernel_size_dim1 * kernel_size_dim1 * kernel_size_dim1)) % kernel_size_dim0)) + 
                128 * kernel_size_dim1 * kernel_size_dim1 * kernel_size_dim1 * (((temp_index // ((-1) + ((-12) * kernel_size_dim1 * kernel_size_dim1) + 6 * kernel_size_dim1 + 8 * kernel_size_dim1 * kernel_size_dim1 * kernel_size_dim1)) % kernel_size_dim0)) + 
                ((temp_index % ((-1) + 2 * kernel_size_dim1))) + 
                (((temp_index // (1 + ((-4) * kernel_size_dim1) + 4 * kernel_size_dim1 * kernel_size_dim1)) % ((-1) + 2 * kernel_size_dim1)))
            ), 
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        temp_zero = tl.where(temp_mask, zero_value, zero_broadcast)

        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        temp_one = tl.where(temp_mask, one_value, one_broadcast)

        temp_mask_broadcast = tl.broadcast_to(temp_mask, [XBLOCK, RBLOCK])
        temp_zero_broadcast = tl.broadcast_to(temp_zero, [XBLOCK, RBLOCK])
        temp_one_broadcast = tl.broadcast_to(temp_one, [XBLOCK, RBLOCK])

        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_combine(
            running_mean, running_m2, running_weight,
            temp_mask_broadcast, temp_zero_broadcast, temp_one_broadcast
        )

        running_mean = tl.where(reduction_mask & input_mask, running_mean_next, running_mean)
        running_m2 = tl.where(reduction_mask & input_mask, running_m2_next, running_m2)
        running_weight = tl.where(reduction_mask & input_mask, running_weight_next, running_weight)

    final_mean, final_variance, final_count = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )

    final_mean = final_mean[:, None]
    final_variance = final_variance[:, None]
    final_count = final_count[:, None]

    tl.store(output_mean_ptr + (input_linear_index), final_mean, input_mask)
    tl.store(output_variance_ptr + (input_linear_index), final_variance, input_mask)
    tl.store(output_count_ptr + (input_linear_index), final_count, input_mask)