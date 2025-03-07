# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_convolution_native_group_norm_0(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 1024
    rnumel = 25200
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_4 = x_indices
    x_indices_0 = x_indices % 8
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_5 = r_indices
        r_indices_3 = (r_indices // 12600)
        
        input_data = tl.load(in_out_ptr0 + (r_indices_5 + (25200 * x_indices_4)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        weight_data = tl.load(in_ptr0 + (r_indices_3 + (2 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        
        combined_data = input_data + weight_data
        broadcasted_data = tl.broadcast_to(combined_data, [XBLOCK, RBLOCK])
        
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_data, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)
        
        tl.store(in_out_ptr0 + (r_indices_5 + (25200 * x_indices_4)), combined_data, r_mask & x_mask)
    
    mean_result, variance_result, weight_result = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    
    tl.store(out_ptr0 + (x_indices_4), mean_result, x_mask)
    
    variance_scale = 25200.0
    variance_normalized = variance_result / variance_scale
    epsilon = 1e-05
    variance_adjusted = variance_normalized + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)
    
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x_indices_4), inv_sqrt_variance, x_mask)