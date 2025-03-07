# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_convolution_native_group_norm_0(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 1024
    rnumel = 1800
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x4 = x_index
    x0 = x_index % 8
    tmp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r5 = r_index
        r3 = (r_index // 900)
        
        input_data = tl.load(in_out_ptr0 + (r5 + (1800 * x4)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        weight_data = tl.load(in_ptr0 + (r3 + (2 * x0)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        
        combined_data = input_data + weight_data
        broadcast_data = tl.broadcast_to(combined_data, [XBLOCK, RBLOCK])
        
        tmp_mean_next, tmp_m2_next, tmp_weight_next = triton_helpers.welford_reduce(
            broadcast_data, tmp_mean, tmp_m2, tmp_weight, r_offset == 0
        )
        
        tmp_mean = tl.where(r_mask & x_mask, tmp_mean_next, tmp_mean)
        tmp_m2 = tl.where(r_mask & x_mask, tmp_m2_next, tmp_m2)
        tmp_weight = tl.where(r_mask & x_mask, tmp_weight_next, tmp_weight)
        
        tl.store(in_out_ptr0 + (r5 + (1800 * x4)), combined_data, r_mask & x_mask)
    
    mean, variance, weight = triton_helpers.welford(tmp_mean, tmp_m2, tmp_weight, 1)
    mean = mean[:, None]
    variance = variance[:, None]
    
    tl.store(out_ptr0 + (x4), mean, x_mask)
    
    scale_factor = 1800.0
    variance_scaled = variance / scale_factor
    epsilon = 1e-05
    variance_scaled_epsilon = variance_scaled + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_scaled_epsilon)
    
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), inv_sqrt_variance, x_mask)