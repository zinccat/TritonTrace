# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_add_hardswish_logsumexp_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 115200
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_index = rindex
    x_mod_900 = xindex % 900
    x_div_900 = (xindex // 900)
    x_full_index = xindex

    input_value0 = tl.load(in_ptr0 + (x_mod_900 + (900 * row_index) + (14400 * x_div_900)), xmask, other=0.0)
    input_value1 = tl.load(in_ptr1 + (x_mod_900 + (900 * row_index) + (14400 * x_div_900)), xmask, other=0.0)
    bias_value = 3.0
    lower_bound = 0.0
    upper_bound = 6.0
    scale_factor = 0.16666666666666666

    biased_input = input_value1 + bias_value
    clamped_input = triton_helpers.minimum(triton_helpers.maximum(biased_input, lower_bound), upper_bound)
    scaled_input = input_value1 * clamped_input
    scaled_and_scaled_input = scaled_input * scale_factor
    combined_input = input_value0 + scaled_and_scaled_input

    broadcasted_combined = tl.broadcast_to(combined_input, [XBLOCK, RBLOCK])
    masked_broadcasted_combined = tl.where(xmask, broadcasted_combined, float("-inf"))

    max_value = triton_helpers.max2(masked_broadcasted_combined, 1)[:, None]
    abs_max_value = tl.math.abs(max_value)
    inf_value = float("inf")
    is_inf = abs_max_value == inf_value
    adjusted_max_value = tl.where(is_inf, lower_bound, max_value)

    stabilized_input = combined_input - adjusted_max_value
    exp_stabilized_input = tl.math.exp(stabilized_input)
    broadcasted_exp = tl.broadcast_to(exp_stabilized_input, [XBLOCK, RBLOCK])
    masked_exp = tl.where(xmask, broadcasted_exp, 0)
    sum_exp = tl.sum(masked_exp, 1)[:, None]
    log_sum_exp = tl.math.log(sum_exp)
    log_sum_exp_stabilized = log_sum_exp + adjusted_max_value

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x_full_index), log_sum_exp_stabilized, xmask)