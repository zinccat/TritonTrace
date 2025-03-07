# From: 99_TripletMarginLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_mean_norm_sub_1(in_out_ptr0, in_ptr0, in_ptr1, ks0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_full_mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < rnumel
        r0_indices = r_indices
        loaded_values0 = tl.load(in_ptr0 + (r0_indices), r_mask, eviction_policy='evict_first', other=0.0)
        loaded_values1 = tl.load(in_ptr1 + (r0_indices), r_mask, eviction_policy='evict_first', other=0.0)
        
        sqrt_values0 = tl.extra.cuda.libdevice.sqrt(loaded_values0)
        constant_one = 1.0
        sum_sqrt_values0 = sqrt_values0 + constant_one
        
        sqrt_values1 = tl.extra.cuda.libdevice.sqrt(loaded_values1)
        difference = sum_sqrt_values0 - sqrt_values1
        
        clamp_min_value = 0.0
        clamped_difference = triton_helpers.maximum(difference, clamp_min_value)
        broadcast_clamped = tl.broadcast_to(clamped_difference, [XBLOCK, RBLOCK])
        
        temp_sum += broadcast_clamped
        temp_sum = tl.where(r_mask, temp_sum, temp_sum)
    
    sum_temp = tl.sum(temp_sum, 1)[:, None]
    ks0_float = ks0.to(tl.float32)
    mean_value = sum_temp / ks0_float
    
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), mean_value, None)