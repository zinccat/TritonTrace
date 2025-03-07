# From: 37_Matmul_Swish_Sum_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_1(
    input_grad_ptr, input_ptr, input_mean_ptr, input_var_ptr, input_inv_std_ptr,
    output_grad_ptr, output_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    input_mean = tl.load(input_mean_ptr + (x3), xmask, eviction_policy='evict_last')
    x1 = xindex // 32
    sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        input_grad = tl.load(input_grad_ptr + (x3 + 1024 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input = tl.load(input_ptr + (x3 + 1024 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_mean_r = tl.load(input_mean_ptr + (x1 + 32 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        input_inv_std_r = tl.load(input_inv_std_ptr + (x1 + 32 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        
        sigmoid_input = tl.sigmoid(input)
        grad_sigmoid = sigmoid_input * input
        grad_mean = grad_sigmoid + input_mean
        grad_input = input_grad * grad_mean
        grad_mean_r = input_grad * input_mean_r
        grad_var = grad_input - grad_mean_r
        grad_var_scaled = grad_var * input_inv_std_r
        grad_var_broadcast = tl.broadcast_to(grad_var_scaled, [XBLOCK, RBLOCK])
        
        sum_grad = sum_grad + grad_var_broadcast
        sum_grad = tl.where(rmask & xmask, sum_grad, sum_grad)
        
        input_broadcast = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
        sum_input = sum_input + input_broadcast
        sum_input = tl.where(rmask & xmask, sum_input, sum_input)
    
    sum_grad_x = tl.sum(sum_grad, 1)[:, None]
    sum_input_x = tl.sum(sum_input, 1)[:, None]
    
    tl.store(output_grad_ptr + (x3), sum_grad_x, xmask)
    tl.store(output_ptr + (x3), sum_input_x, xmask)