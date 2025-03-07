# From: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_neg_sum_2(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_sum_negatives = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum_originals = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_flat = r_indices
        
        loaded_values = tl.load(in_ptr0 + (x_indices_flat + 512 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        negated_values = -loaded_values
        broadcast_negated = tl.broadcast_to(negated_values, [XBLOCK, RBLOCK])
        temp_sum_negatives = temp_sum_negatives + broadcast_negated
        temp_sum_negatives = tl.where(r_mask & x_mask, temp_sum_negatives, temp_sum_negatives)
        
        broadcast_original = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_sum_originals = temp_sum_originals + broadcast_original
        temp_sum_originals = tl.where(r_mask & x_mask, temp_sum_originals, temp_sum_originals)
    
    summed_negatives = tl.sum(temp_sum_negatives, 1)[:, None]
    summed_originals = tl.sum(temp_sum_originals, 1)[:, None]
    
    tl.store(out_ptr0 + (x_indices_flat), summed_negatives, x_mask)
    tl.store(out_ptr1 + (x_indices_flat), summed_originals, x_mask)