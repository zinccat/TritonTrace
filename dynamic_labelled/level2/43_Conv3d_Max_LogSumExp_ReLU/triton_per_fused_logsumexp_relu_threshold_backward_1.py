# From: 43_Conv3d_Max_LogSumExp_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_logsumexp_relu_threshold_backward_1(
    input_ptr, output_ptr_max, output_ptr_sum_exp, output_ptr_threshold, 
    kernel_size_0, kernel_size_1, kernel_size_2, input_num_elements, 
    reduction_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_offset = reduction_index
    kernel_index_0 = (input_index % kernel_size_0)
    kernel_index_1 = input_index // kernel_size_0
    linear_index = input_index
    loaded_input = tl.load(
        input_ptr + (
            kernel_index_0 + reduction_offset * (kernel_size_2 // 2) * (kernel_size_2 // 2) * (kernel_size_1 // 2) 
            + 16 * kernel_index_1 * (kernel_size_2 // 2) * (kernel_size_2 // 2) * (kernel_size_1 // 2)
        ), 
        input_mask, 
        eviction_policy='evict_last', 
        other=0.0
    )
    broadcasted_input = tl.broadcast_to(loaded_input, [XBLOCK, RBLOCK])
    masked_input = tl.where(input_mask, broadcasted_input, float("-inf"))
    max_values = triton_helpers.max2(masked_input, 1)[:, None]
    abs_max_values = tl.math.abs(max_values)
    inf_value = float("inf")
    is_inf = abs_max_values == inf_value
    zero_value = 0.0
    adjusted_max_values = tl.where(is_inf, zero_value, max_values)
    centered_input = loaded_input - adjusted_max_values
    exp_centered_input = tl.math.exp(centered_input)
    broadcasted_exp = tl.broadcast_to(exp_centered_input, [XBLOCK, RBLOCK])
    masked_exp = tl.where(input_mask, broadcasted_exp, 0)
    sum_exp = tl.sum(masked_exp, 1)[:, None]
    log_sum_exp = tl.math.log(sum_exp)
    log_sum_exp_adjusted = log_sum_exp + adjusted_max_values
    zero_int32 = tl.full([1, 1], 0, tl.int32)
    max_log_sum_exp = triton_helpers.maximum(zero_int32, log_sum_exp_adjusted)
    threshold_mask = max_log_sum_exp <= zero_value
    tl.store(output_ptr_threshold + (linear_index), max_log_sum_exp, input_mask)
    tl.store(output_ptr_max + (linear_index), adjusted_max_values, input_mask)
    tl.store(output_ptr_sum_exp + (linear_index), sum_exp, input_mask)
    tl.store(output_ptr_threshold + (linear_index), threshold_mask, input_mask)