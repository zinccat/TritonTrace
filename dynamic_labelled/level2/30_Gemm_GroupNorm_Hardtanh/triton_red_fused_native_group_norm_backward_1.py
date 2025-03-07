# From: 30_Gemm_GroupNorm_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_1(
    input_grad_ptr, input_ptr, input_mean_ptr, input_inv_std_ptr, input_weight_ptr, 
    output_grad_ptr, output_mean_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 64
    sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_weighted_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        
        grad_input = tl.load(input_grad_ptr + (x3 + 512 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_data = tl.load(input_ptr + (x3 + 512 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_grad = tl.load(input_mean_ptr + (x3 + 512 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_weight = tl.load(input_weight_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        input_inv_std = tl.load(input_inv_std_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        
        lower_bound = -2.0
        upper_bound = 2.0
        is_out_of_bounds = (grad_input <= lower_bound) | (grad_input >= upper_bound)
        clamped_input = tl.where(is_out_of_bounds, 0.0, input_data)
        
        weighted_input = clamped_input * input_grad
        weighted_input_diff = weighted_input - (clamped_input * input_weight)
        weighted_input_scaled = weighted_input_diff * input_inv_std
        
        broadcast_weighted_input_scaled = tl.broadcast_to(weighted_input_scaled, [XBLOCK, RBLOCK])
        sum_grad = sum_grad + broadcast_weighted_input_scaled
        sum_grad = tl.where(rmask & xmask, sum_grad, sum_grad)
        
        broadcast_clamped_input = tl.broadcast_to(clamped_input, [XBLOCK, RBLOCK])
        sum_weighted_input = sum_weighted_input + broadcast_clamped_input
        sum_weighted_input = tl.where(rmask & xmask, sum_weighted_input, sum_weighted_input)
    
    output_grad_sum = tl.sum(sum_grad, 1)[:, None]
    output_mean_sum = tl.sum(sum_weighted_input, 1)[:, None]
    
    tl.store(output_grad_ptr + (x3), output_grad_sum, xmask)
    tl.store(output_mean_ptr + (x3), output_mean_sum, xmask)