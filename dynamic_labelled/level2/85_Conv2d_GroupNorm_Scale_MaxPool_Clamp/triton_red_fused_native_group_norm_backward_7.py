# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_7(
    input_grad_ptr, input_ptr, mean_ptr, inv_std_ptr, output_grad_ptr, output_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 2
    sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        grad_input = tl.load(input_grad_ptr + (x3 + 16 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_value = tl.load(input_ptr + (x3 + 16 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean_value = tl.load(mean_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        inv_std_value = tl.load(inv_std_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        
        input_mean_product = input_value * mean_value
        grad_centered = grad_input - input_mean_product
        grad_scaled = grad_centered * inv_std_value
        grad_scaled_broadcast = tl.broadcast_to(grad_scaled, [XBLOCK, RBLOCK])
        
        sum_grad_temp = sum_grad + grad_scaled_broadcast
        sum_grad = tl.where(rmask & xmask, sum_grad_temp, sum_grad)
        
        input_broadcast = tl.broadcast_to(input_value, [XBLOCK, RBLOCK])
        sum_input_temp = sum_input + input_broadcast
        sum_input = tl.where(rmask & xmask, sum_input_temp, sum_input)
    
    output_grad_sum = tl.sum(sum_grad, 1)[:, None]
    output_input_sum = tl.sum(sum_input, 1)[:, None]
    
    tl.store(output_grad_ptr + (x3), output_grad_sum, xmask)
    tl.store(output_ptr + (x3), output_input_sum, xmask)