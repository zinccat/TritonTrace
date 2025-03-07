# From: 37_Matmul_Swish_Sum_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_1(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, 
    output_grad_ptr, output_ptr, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    mean = tl.load(mean_ptr + (x3), xmask, eviction_policy='evict_last')
    x1 = xindex // 32
    sum_grad_gamma = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_grad_x = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        grad_input = tl.load(input_grad_ptr + (x3 + 1024 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input = tl.load(input_ptr + (x3 + 1024 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean_r = tl.load(mean_ptr + (x1 + 32 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        variance_r = tl.load(variance_ptr + (x1 + 32 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        
        sigmoid_input = tl.sigmoid(input)
        grad_input_scaled = sigmoid_input * input
        grad_input_scaled_sum = grad_input_scaled + mean
        grad_gamma = grad_input * grad_input_scaled_sum
        grad_mean = grad_input * mean_r
        grad_variance = grad_gamma - grad_mean
        grad_variance_scaled = grad_variance * variance_r
        grad_variance_broadcast = tl.broadcast_to(grad_variance_scaled, [XBLOCK, RBLOCK])
        
        sum_grad_gamma = sum_grad_gamma + grad_variance_broadcast
        sum_grad_gamma = tl.where(rmask & xmask, sum_grad_gamma, sum_grad_gamma)
        
        grad_input_broadcast = tl.broadcast_to(grad_input, [XBLOCK, RBLOCK])
        sum_grad_x = sum_grad_x + grad_input_broadcast
        sum_grad_x = tl.where(rmask & xmask, sum_grad_x, sum_grad_x)
    
    sum_grad_gamma_final = tl.sum(sum_grad_gamma, 1)[:, None]
    sum_grad_x_final = tl.sum(sum_grad_x, 1)[:, None]
    
    tl.store(output_grad_ptr + (x3), sum_grad_gamma_final, xmask)
    tl.store(output_ptr + (x3), sum_grad_x_final, xmask)