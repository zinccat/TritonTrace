# From: 43_Conv3d_Max_LogSumExp_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_logsumexp_relu_threshold_backward_1(
    input_ptr, output_ptr_max, output_ptr_sum_exp, output_ptr_threshold, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = x_indices % 2048
    x1 = (x_indices // 2048)
    x3 = x_indices
    input_values = tl.load(input_ptr + (x0 + (2048 * r2) + (32768 * x1)), None)
    broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    max_values = triton_helpers.max2(broadcasted_values, 1)[:, None]
    abs_max_values = tl.math.abs(max_values)
    inf_value = float("inf")
    is_inf = abs_max_values == inf_value
    zero_value = 0.0
    adjusted_max_values = tl.where(is_inf, zero_value, max_values)
    adjusted_input_values = input_values - adjusted_max_values
    exp_values = tl.math.exp(adjusted_input_values)
    broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    sum_exp_values = tl.sum(broadcasted_exp_values, 1)[:, None]
    log_sum_exp_values = tl.math.log(sum_exp_values)
    final_values = log_sum_exp_values + adjusted_max_values
    zero_int32 = tl.full([1, 1], 0, tl.int32)
    max_final_values = triton_helpers.maximum(zero_int32, final_values)
    threshold_mask = max_final_values <= zero_value
    tl.store(output_ptr_max + (x3), max_final_values, None)
    tl.store(output_ptr_threshold + (x3), threshold_mask, None)
    tl.store(output_ptr_max + (x3), max_values, None)
    tl.store(output_ptr_sum_exp + (x3), sum_exp_values, None)