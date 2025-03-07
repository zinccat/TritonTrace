# From: 31_VisionAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_5red_fused__softmax_5(in_out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    rnumel = 16384
    x_offset = tl.program_id(0).to(tl.int64) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :].to(tl.int64)
    x0 = x_indices
    max_values = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r1 = r_indices
        loaded_values = tl.load(in_out_ptr0 + (r1 + 16384 * x0), r_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        max_values = triton_helpers.maximum(max_values, broadcasted_values)
        max_values = tl.where(r_mask, max_values, max_values)
    
    max_per_row = triton_helpers.max2(max_values, 1)[:, None]
    sum_exp_values = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r1 = r_indices
        loaded_values = tl.load(in_out_ptr0 + (r1 + 16384 * x0), r_mask, eviction_policy='evict_last', other=0.0)
        adjusted_values = loaded_values - max_per_row
        exp_values = tl.math.exp(adjusted_values)
        broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
        sum_exp_values = sum_exp_values + broadcasted_exp_values
        sum_exp_values = tl.where(r_mask, sum_exp_values, sum_exp_values)
    
    sum_exp_per_row = tl.sum(sum_exp_values, 1)[:, None]
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r1 = r_indices
        loaded_values = tl.load(in_out_ptr0 + (r1 + 16384 * x0), r_mask, eviction_policy='evict_first', other=0.0)
        adjusted_values = loaded_values - max_per_row
        exp_values = tl.math.exp(adjusted_values)
        softmax_values = exp_values / sum_exp_per_row
        tl.store(in_out_ptr0 + (r1 + 16384 * x0), softmax_values, r_mask)