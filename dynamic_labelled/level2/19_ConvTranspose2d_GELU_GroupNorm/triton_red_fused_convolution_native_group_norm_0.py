# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_native_group_norm_0(
    in_out_ptr, input_ptr, output_mean_ptr, output_var_ptr, output_inv_std_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    rnumel = 34848
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x4 = x_index
    x0 = (x_index % 8)
    
    tmp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r5 = r_index
        r3 = r_index // 4356
        
        loaded_in_out = tl.load(in_out_ptr + (r5 + 34848 * x4), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        loaded_input = tl.load(input_ptr + (r3 + 8 * x0), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        
        tmp_sum = loaded_in_out + loaded_input
        tmp_half = 0.5
        tmp_scaled = tmp_sum * tmp_half
        tmp_sqrt2_inv = 0.7071067811865476
        tmp_erf_input = tmp_sum * tmp_sqrt2_inv
        tmp_erf = tl.extra.cuda.libdevice.erf(tmp_erf_input)
        tmp_one = 1.0
        tmp_erf_scaled = tmp_scaled * (tmp_erf + tmp_one)
        tmp_broadcast = tl.broadcast_to(tmp_erf_scaled, [XBLOCK, RBLOCK])
        
        tmp_mean_next, tmp_m2_next, tmp_weight_next = triton_helpers.welford_reduce(
            tmp_broadcast, tmp_mean, tmp_m2, tmp_weight, r_offset == 0
        )
        
        tmp_mean = tl.where(r_mask & x_mask, tmp_mean_next, tmp_mean)
        tmp_m2 = tl.where(r_mask & x_mask, tmp_m2_next, tmp_m2)
        tmp_weight = tl.where(r_mask & x_mask, tmp_weight_next, tmp_weight)
        
        tl.store(in_out_ptr + (r5 + 34848 * x4), tmp_sum, r_mask & x_mask)
    
    tmp_mean_final, tmp_m2_final, tmp_weight_final = triton_helpers.welford(
        tmp_mean, tmp_m2, tmp_weight, 1
    )
    
    tmp_mean_final = tmp_mean_final[:, None]
    tmp_m2_final = tmp_m2_final[:, None]
    
    tl.store(output_mean_ptr + (x4), tmp_mean_final, x_mask)
    tl.store(output_var_ptr + (x4), tmp_m2_final, x_mask)
    
    tmp_rnumel = 34848.0
    tmp_var = tmp_m2_final / tmp_rnumel
    tmp_epsilon = 1e-05
    tmp_var_eps = tmp_var + tmp_epsilon
    tmp_inv_std = tl.extra.cuda.libdevice.rsqrt(tmp_var_eps)
    
    tl.store(output_inv_std_ptr + (x4), tmp_inv_std, x_mask)