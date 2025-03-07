# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_add_clamp_div_logsumexp_max_mul_sigmoid_sub_1(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 15748992
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x2 = (xindex // 123039)
    x5 = xindex % 123039
    x0 = xindex % 3969
    x4 = (xindex // 3969)
    x6 = xindex

    # Load input data with masking
    input_data = tl.load(in_ptr0 + (x5 + (123039 * r3) + (1968624 * x2)), xmask, other=0.0)
    additional_input = tl.load(in_ptr1 + (r3), None, eviction_policy='evict_last')

    # Broadcast and mask operations
    broadcasted_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
    masked_input = tl.where(xmask, broadcasted_input, float("-inf"))

    # Compute max and handle infinities
    max_values = triton_helpers.max2(masked_input, 1)[:, None]
    abs_max_values = tl.math.abs(max_values)
    inf_mask = abs_max_values == float("inf")
    safe_max_values = tl.where(inf_mask, 0.0, max_values)

    # Compute exponentials and log-sum-exp
    adjusted_input = input_data - safe_max_values
    exp_values = tl.math.exp(adjusted_input)
    broadcasted_exp = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp = tl.where(xmask, broadcasted_exp, 0)
    sum_exp = tl.sum(masked_exp, 1)[:, None]
    log_sum_exp = tl.math.log(sum_exp)
    log_sum_exp_adjusted = log_sum_exp + safe_max_values

    # Sigmoid and final computation
    sigmoid_input = log_sum_exp_adjusted + 3.0
    sigmoid_output = tl.sigmoid(sigmoid_input)
    weighted_log_sum_exp = log_sum_exp_adjusted * sigmoid_output
    final_output = weighted_log_sum_exp * 0.16666666666666666
    result = final_output - additional_input

    # Clamp result between -1 and 1
    clamped_result = triton_helpers.maximum(result, -1.0)
    clamped_result = triton_helpers.minimum(clamped_result, 1.0)
    broadcasted_clamped = tl.broadcast_to(clamped_result, [XBLOCK, RBLOCK])
    masked_clamped = tl.where(xmask, broadcasted_clamped, float("-inf"))

    # Find max and index
    max_clamped = triton_helpers.max2(masked_clamped, 1)[:, None]
    max_index = triton_helpers.max_with_index(masked_clamped, tl.broadcast_to(rindex, masked_clamped.shape), 1)[:, None]

    # Store results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0 + (4000 * x4)), log_sum_exp_adjusted, xmask)
    tl.store(out_ptr1 + (x6), max_clamped, xmask)
    tl.store(out_ptr2 + (x0 + (3984 * x4)), max_index, xmask)