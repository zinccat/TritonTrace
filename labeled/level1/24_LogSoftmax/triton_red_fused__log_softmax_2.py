# From: 24_LogSoftmax

import triton
import triton.language as tl


@triton.jit
def triton_red_fused__log_softmax_2(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 32
    rnumel = 8192
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_3 = x_indices
    x_indices_1 = (x_indices // 2)
    
    input_values_1 = tl.load(in_ptr1 + (x_indices_1), x_mask, eviction_policy='evict_last')
    sum_exp_values = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_2 = r_indices
        
        input_values_0 = tl.load(in_ptr0 + (r_indices_2 + (8192 * x_indices_3)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        max_subtracted_values = input_values_0 - input_values_1
        exp_values = tl.math.exp(max_subtracted_values)
        broadcast_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
        
        sum_exp_values = tl.where(r_mask & x_mask, sum_exp_values + broadcast_exp_values, sum_exp_values)
    
    sum_exp_per_row = tl.sum(sum_exp_values, 1)[:, None]
    tl.store(out_ptr0 + (x_indices_3), sum_exp_per_row, x_mask)