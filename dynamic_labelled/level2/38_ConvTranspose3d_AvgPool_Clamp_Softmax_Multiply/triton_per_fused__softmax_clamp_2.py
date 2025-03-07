# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_clamp_2per_fused__softmax_clamp_2(
    input_ptr, output_ptr_max, output_ptr_sum, kernel_size_0, kernel_size_1, kernel_size_2, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_indices_2 = reduction_indices
    input_mod_0 = (input_indices % kernel_size_0)
    input_div_0 = input_indices // kernel_size_0
    input_flat_index = input_indices
    loaded_values = tl.load(
        input_ptr + (input_mod_0 + kernel_size_1 * kernel_size_2 * reduction_indices_2 + 16 * kernel_size_1 * kernel_size_2 * input_div_0), 
        input_mask, 
        eviction_policy='evict_last', 
        other=0.0
    )
    zero_value = 0.0
    max_values = triton_helpers.maximum(loaded_values, zero_value)
    clamp_max_value = 1.0
    clamped_values = triton_helpers.minimum(max_values, clamp_max_value)
    broadcast_clamped = tl.broadcast_to(clamped_values, [XBLOCK, RBLOCK])
    masked_clamped_values = tl.where(input_mask, broadcast_clamped, float("-inf"))
    max_across_reductions = triton_helpers.max2(masked_clamped_values, 1)[:, None]
    shifted_values = clamped_values - max_across_reductions
    exp_values = tl.math.exp(shifted_values)
    broadcast_exp = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp_values = tl.where(input_mask, broadcast_exp, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]
    tl.store(output_ptr_max + (input_flat_index), max_across_reductions, input_mask)
    tl.store(output_ptr_sum + (input_flat_index), sum_exp_values, input_mask)