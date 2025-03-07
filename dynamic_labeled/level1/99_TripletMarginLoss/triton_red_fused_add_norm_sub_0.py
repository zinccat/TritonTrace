# From: 99_TripletMarginLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_norm_sub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ks0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    sum_squares_diff1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_squares_diff2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        data0 = tl.load(in_ptr0 + (r1 + ks0 * x0), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        data1 = tl.load(in_ptr1 + (r1 + ks0 * x0), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        data2 = tl.load(in_ptr2 + (r1 + ks0 * x0), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        
        diff1 = data0 - data1
        epsilon = 1e-06
        diff1_adjusted = diff1 + epsilon
        squared_diff1 = diff1_adjusted * diff1_adjusted
        broadcasted_squared_diff1 = tl.broadcast_to(squared_diff1, [XBLOCK, RBLOCK])
        sum_squares_diff1 = sum_squares_diff1 + broadcasted_squared_diff1
        sum_squares_diff1 = tl.where(r_mask & x_mask, sum_squares_diff1, sum_squares_diff1)
        
        diff2 = data0 - data2
        diff2_adjusted = diff2 + epsilon
        squared_diff2 = diff2_adjusted * diff2_adjusted
        broadcasted_squared_diff2 = tl.broadcast_to(squared_diff2, [XBLOCK, RBLOCK])
        sum_squares_diff2 = sum_squares_diff2 + broadcasted_squared_diff2
        sum_squares_diff2 = tl.where(r_mask & x_mask, sum_squares_diff2, sum_squares_diff2)
    
    sum_over_r_diff1 = tl.sum(sum_squares_diff1, 1)[:, None]
    sum_over_r_diff2 = tl.sum(sum_squares_diff2, 1)[:, None]
    
    tl.store(out_ptr0 + (x0), sum_over_r_diff1, x_mask)
    tl.store(out_ptr1 + (x0), sum_over_r_diff2, x_mask)