# From: 68_Matmul_Min_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_div_eq_gt_lt_masked_fill_neg_sum_where_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    rmask_full = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase_indices = tl.arange(0, RBLOCK)[None, :]
    neg_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    threshold_value = tl.load(in_ptr2 + (0))
    threshold_broadcast = tl.broadcast_to(threshold_value, [XBLOCK, RBLOCK])
    sum_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase_indices
        rmask = rindex < rnumel
        r0 = rindex
        input0_values = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        input1_values = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        neg_input0 = -input0_values
        neg_input0_broadcast = tl.broadcast_to(neg_input0, [XBLOCK, RBLOCK])
        neg_accumulator_update = neg_accumulator + neg_input0_broadcast
        neg_accumulator = tl.where(rmask, neg_accumulator_update, neg_accumulator)

        less_than_threshold = input1_values < threshold_broadcast
        equal_to_threshold = input1_values == threshold_broadcast
        half_value = 0.5
        half_input0 = input0_values * half_value
        conditional_input0 = tl.where(equal_to_threshold, half_input0, input0_values)
        zero_value = 0.0
        conditional_input1 = tl.where(less_than_threshold, zero_value, conditional_input0)
        conditional_input1_broadcast = tl.broadcast_to(conditional_input1, [XBLOCK, RBLOCK])
        sum_accumulator_update = sum_accumulator + conditional_input1_broadcast
        sum_accumulator = tl.where(rmask, sum_accumulator_update, sum_accumulator)

        greater_than_threshold = input1_values > threshold_broadcast
        conditional_output = tl.where(greater_than_threshold, zero_value, conditional_input0)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), conditional_output, rmask)

    neg_sum = tl.sum(neg_accumulator, 1)[:, None]
    sum_result = tl.sum(sum_accumulator, 1)[:, None]
    final_result = neg_sum + sum_result
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), final_result, None)