# From: 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_clamp_ge_le_logical_and_mean_1(
    input_ptr, output_ptr_clamp, output_ptr_logical, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices_0 = input_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_indices_1 = reduction_indices

        temp_load = tl.load(
            input_ptr + (reduction_indices_1 + ((-1) * input_indices_0) + kernel_size_0 * input_indices_0 + 
                         ((-1) * input_indices_0 * kernel_size_1 * kernel_size_1) + 
                         2 * kernel_size_1 * input_indices_0 + 
                         kernel_size_0 * input_indices_0 * kernel_size_1 * kernel_size_1 + 
                         ((-2) * kernel_size_0 * kernel_size_1 * input_indices_0)),
            reduction_mask & input_mask, eviction_policy='evict_first', other=0.0
        )
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_accumulate = temp_sum + temp_broadcast
        temp_sum = tl.where(reduction_mask & input_mask, temp_accumulate, temp_sum)

    temp_mean = tl.sum(temp_sum, 1)[:, None]
    kernel_size_0_float = kernel_size_0.to(tl.float32)
    kernel_size_1_float = kernel_size_1.to(tl.float32)
    kernel_size_2_float = kernel_size_2.to(tl.float32)
    kernel_size_3_float = kernel_size_3.to(tl.float32)

    denominator = (-1) + kernel_size_0 + ((-1) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + ((-2) * kernel_size_0 * kernel_size_1)
    denominator_float = denominator.to(tl.float32)

    mean_value = temp_mean / denominator_float
    clamped_value = triton_helpers.maximum(mean_value, kernel_size_2_float)
    clamped_value = triton_helpers.minimum(clamped_value, kernel_size_3_float)

    greater_equal_mask = mean_value >= kernel_size_2_float
    less_equal_mask = mean_value <= kernel_size_3_float
    logical_and_mask = greater_equal_mask & less_equal_mask

    tl.store(output_ptr_clamp + (input_indices_0), clamped_value, input_mask)
    tl.store(output_ptr_logical + (input_indices_0), logical_and_mask, input_mask)