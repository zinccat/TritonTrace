# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_1(
    input_grad_ptr, mean_ptr, inv_std_ptr, grad_output_ptr, output_grad_mean_ptr, output_grad_var_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 8
    temp_sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        grad_input = tl.load(input_grad_ptr + (x3 + 64 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        grad_output = tl.load(grad_output_ptr + (x3 + 64 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean = tl.load(mean_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        inv_std = tl.load(inv_std_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        
        input_grad_mean = grad_output * mean
        grad_input_centered = grad_input - input_grad_mean
        grad_input_scaled = grad_input_centered * inv_std
        grad_input_scaled_broadcast = tl.broadcast_to(grad_input_scaled, [XBLOCK, RBLOCK])
        
        temp_sum_grad += grad_input_scaled_broadcast
        temp_sum_grad = tl.where(rmask & xmask, temp_sum_grad, temp_sum_grad)
        
        grad_output_broadcast = tl.broadcast_to(grad_output, [XBLOCK, RBLOCK])
        temp_sum_input += grad_output_broadcast
        temp_sum_input = tl.where(rmask & xmask, temp_sum_input, temp_sum_input)
    
    sum_grad = tl.sum(temp_sum_grad, 1)[:, None]
    sum_input = tl.sum(temp_sum_input, 1)[:, None]
    
    tl.store(output_grad_mean_ptr + (x3), sum_grad, xmask)
    tl.store(output_grad_var_ptr + (x3), sum_input, xmask)