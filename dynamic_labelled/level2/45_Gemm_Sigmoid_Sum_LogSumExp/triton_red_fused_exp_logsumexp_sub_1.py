# From: 45_Gemm_Sigmoid_Sum_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_exp_logsumexp_sub_1(in_out_ptr0, in_out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_base = tl.arange(0, RBLOCK)[None, :]
    max_values = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_indices = r_index
        loaded_values = tl.load(in_out_ptr1 + (r_indices), r_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        max_values = triton_helpers.maximum(max_values, broadcasted_values)
        max_values = tl.where(r_mask, max_values, max_values)

    max_across_rows = triton_helpers.max2(max_values, 1)[:, None]
    exp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_indices = r_index
        loaded_values = tl.load(in_out_ptr1 + (r_indices), r_mask, eviction_policy='evict_last', other=0.0)
        abs_max = tl.math.abs(max_across_rows)
        inf_value = float("inf")
        is_inf = abs_max == inf_value
        zero_value = 0.0
        adjusted_max = tl.where(is_inf, zero_value, max_across_rows)
        adjusted_values = loaded_values - adjusted_max
        exp_values = tl.math.exp(adjusted_values)
        broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
        exp_sum = exp_sum + broadcasted_exp_values
        exp_sum = tl.where(r_mask, exp_sum, exp_sum)

    log_sum_exp = tl.math.log(tl.sum(exp_sum, 1)[:, None])
    abs_max = tl.math.abs(max_across_rows)
    is_inf = abs_max == inf_value
    adjusted_max = tl.where(is_inf, zero_value, max_across_rows)
    result = log_sum_exp + adjusted_max

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), result, None)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_indices = r_index
        loaded_values = tl.load(in_out_ptr1 + (r_indices), r_mask, eviction_policy='evict_first', other=0.0)
        adjusted_values = loaded_values - result
        exp_values = tl.math.exp(adjusted_values)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r_indices, [XBLOCK, RBLOCK])), exp_values, r_mask)