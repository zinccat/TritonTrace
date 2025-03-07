# From: 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_clamp_ge_le_logical_and_mean_1red_fused_clamp_ge_le_logical_and_mean_1(
    input_ptr, output_ptr_clamped, output_ptr_mask, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices_0 = input_indices
    accumulated_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_indices_1 = reduction_indices

        load_indices = (
            reduction_indices_1 
            + ((-1) * input_indices_0) 
            + kernel_size_0 * input_indices_0 
            + ((-1) * input_indices_0 * kernel_size_1 * kernel_size_1) 
            + 2 * kernel_size_1 * input_indices_0 
            + kernel_size_0 * input_indices_0 * kernel_size_1 * kernel_size_1 
            + ((-2) * kernel_size_0 * kernel_size_1 * input_indices_0)
        )

        loaded_values = tl.load(input_ptr + load_indices, reduction_mask & input_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        accumulated_sum += broadcasted_values
        accumulated_sum = tl.where(reduction_mask & input_mask, accumulated_sum, accumulated_sum)

    summed_values = tl.sum(accumulated_sum, 1)[:, None]
    normalization_factor = (
        (-1) + kernel_size_0 + ((-1) * kernel_size_1 * kernel_size_1) 
        + 2 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 
        + ((-2) * kernel_size_0 * kernel_size_1)
    ).to(tl.float32)

    mean_values = summed_values / normalization_factor
    clamp_min = kernel_size_2.to(tl.float32)
    clamped_values = triton_helpers.maximum(mean_values, clamp_min)
    clamp_max = kernel_size_3.to(tl.float32)
    clamped_values = triton_helpers.minimum(clamped_values, clamp_max)

    greater_equal_mask = mean_values >= clamp_min
    less_equal_mask = mean_values <= clamp_max
    logical_and_mask = greater_equal_mask & less_equal_mask

    tl.store(output_ptr_clamped + (input_indices_0), clamped_values, input_mask)
    tl.store(output_ptr_mask + (input_indices_0), logical_and_mask, input_mask)