# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_3(
    input_grad_ptr, mean_ptr, inv_std_ptr, input_ptr, output_grad_ptr, mean_grad_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 16
    sum_grad_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        grad_input = tl.load(input_grad_ptr + (x3 + 128 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        grad_output = tl.load(mean_ptr + (x3 + 128 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean = tl.load(mean_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        inv_std = tl.load(inv_std_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        
        grad_input_scaled = grad_output * mean
        grad_input_centered = grad_input - grad_input_scaled
        grad_input_normalized = grad_input_centered * inv_std
        grad_input_broadcast = tl.broadcast_to(grad_input_normalized, [XBLOCK, RBLOCK])
        
        sum_grad_input += grad_input_broadcast
        sum_grad_input = tl.where(rmask & xmask, sum_grad_input, sum_grad_input)
        
        input_broadcast = tl.broadcast_to(grad_output, [XBLOCK, RBLOCK])
        sum_input += input_broadcast
        sum_input = tl.where(rmask & xmask, sum_input, sum_input)
    
    grad_input_sum = tl.sum(sum_grad_input, 1)[:, None]
    input_sum = tl.sum(sum_input, 1)[:, None]
    
    tl.store(output_grad_ptr + (x3), grad_input_sum, xmask)
    tl.store(mean_grad_ptr + (x3), input_sum, xmask)