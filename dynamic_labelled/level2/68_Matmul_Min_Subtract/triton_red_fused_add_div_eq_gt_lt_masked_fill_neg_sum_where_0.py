# From: 68_Matmul_Min_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_div_eq_gt_lt_masked_fill_neg_sum_where_0(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_base = tl.arange(0, RBLOCK)[None, :]
    zero_matrix = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp_value_from_in_ptr2 = tl.load(in_ptr2 + (0))
    broadcast_tmp_value = tl.broadcast_to(tmp_value_from_in_ptr2, [XBLOCK, RBLOCK])
    accumulated_negatives = zero_matrix
    accumulated_results = zero_matrix

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r0 = r_index
        loaded_values0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        loaded_values1 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        negated_values0 = -loaded_values0
        broadcast_negated_values = tl.broadcast_to(negated_values0, [XBLOCK, RBLOCK])
        updated_accumulated_negatives = accumulated_negatives + broadcast_negated_values
        accumulated_negatives = tl.where(r_mask, updated_accumulated_negatives, accumulated_negatives)
        
        less_than_mask = loaded_values1 < broadcast_tmp_value
        equal_mask = loaded_values1 == broadcast_tmp_value
        half_value = 0.5
        half_of_values0 = loaded_values0 * half_value
        conditional_values = tl.where(equal_mask, half_of_values0, loaded_values0)
        zero_value = 0.0
        final_values = tl.where(less_than_mask, zero_value, conditional_values)
        broadcast_final_values = tl.broadcast_to(final_values, [XBLOCK, RBLOCK])
        updated_accumulated_results = accumulated_results + broadcast_final_values
        accumulated_results = tl.where(r_mask, updated_accumulated_results, accumulated_results)
        
        greater_than_mask = loaded_values1 > broadcast_tmp_value
        conditional_greater_values = tl.where(greater_than_mask, zero_value, conditional_values)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), conditional_greater_values, r_mask)

    sum_of_negatives = tl.sum(accumulated_negatives, 1)[:, None]
    sum_of_results = tl.sum(accumulated_results, 1)[:, None]
    total_sum = sum_of_negatives + sum_of_results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), total_sum, None)