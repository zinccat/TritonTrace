# From: 96_HuberLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_smooth_l1_loss_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r1 = r_indices
        combined_index = r1 + x0 * ((63 + ks0 * ks1) // 64)
        max_index = ks0 * ks1
        valid_index_mask = combined_index < max_index
        
        value0 = tl.load(in_ptr0 + ((combined_index % max_index)), r_mask & valid_index_mask & x_mask, eviction_policy='evict_last', other=0.0)
        value1 = tl.load(in_ptr1 + ((combined_index % max_index)), r_mask & valid_index_mask & x_mask, eviction_policy='evict_last', other=0.0)
        
        diff = value0 - value1
        abs_diff = tl.math.abs(diff)
        threshold = 1.0
        below_threshold = abs_diff < threshold
        
        squared_diff = abs_diff * abs_diff
        half = 0.5
        smooth_l1 = squared_diff * half * threshold
        l1 = abs_diff - half
        
        smooth_l1_loss = tl.where(below_threshold, smooth_l1, l1)
        broadcast_loss = tl.full(smooth_l1_loss.shape, 0, smooth_l1_loss.dtype)
        masked_loss = tl.where(valid_index_mask, smooth_l1_loss, broadcast_loss)
        expanded_loss = tl.broadcast_to(masked_loss, [XBLOCK, RBLOCK])
        
        temp_sum += expanded_loss
        temp_sum = tl.where(r_mask & x_mask, temp_sum, temp_sum)
    
    reduced_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (x0), reduced_sum, x_mask)