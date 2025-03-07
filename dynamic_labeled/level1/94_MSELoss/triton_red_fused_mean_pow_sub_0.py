# From: 94_MSELoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_pow_sub_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        combined_index = r1 + x0 * ((63 + ks0 * ks1) // 64)
        total_elements = ks0 * ks1
        index_within_bounds = combined_index < total_elements
        
        value0 = tl.load(in_ptr0 + ((combined_index % total_elements)), index_within_bounds & x_mask, eviction_policy='evict_last', other=0.0)
        value1 = tl.load(in_ptr1 + ((combined_index % total_elements)), index_within_bounds & x_mask, eviction_policy='evict_last', other=0.0)
        
        difference = value0 - value1
        squared_difference = difference * difference
        zero_filled = tl.full(squared_difference.shape, 0, squared_difference.dtype)
        
        selected_values = tl.where(index_within_bounds, squared_difference, zero_filled)
        broadcasted_values = tl.broadcast_to(selected_values, [XBLOCK, RBLOCK])
        
        temp_sum += tl.where(r_mask & x_mask, broadcasted_values, temp_sum)
    
    reduced_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (x0), reduced_sum, x_mask)