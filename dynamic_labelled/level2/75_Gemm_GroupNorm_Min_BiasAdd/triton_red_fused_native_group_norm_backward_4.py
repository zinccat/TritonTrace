# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_4(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, output_grad_ptr, output_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 32
    sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        grad_input = tl.load(input_grad_ptr + (x3 + 256 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_value = tl.load(input_ptr + (x3 + 256 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean_value = tl.load(mean_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        variance_value = tl.load(variance_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        
        grad_input_times_input = grad_input * input_value
        grad_input_times_mean = grad_input * mean_value
        grad_input_minus_mean = grad_input_times_input - grad_input_times_mean
        grad_output = grad_input_minus_mean * variance_value
        
        broadcast_grad_output = tl.broadcast_to(grad_output, [XBLOCK, RBLOCK])
        sum_grad_temp = sum_grad + broadcast_grad_output
        sum_grad = tl.where(rmask & xmask, sum_grad_temp, sum_grad)
        
        broadcast_input = tl.broadcast_to(input_value, [XBLOCK, RBLOCK])
        sum_input_temp = sum_input + broadcast_input
        sum_input = tl.where(rmask & xmask, sum_input_temp, sum_input)
    
    output_grad_sum = tl.sum(sum_grad, 1)[:, None]
    output_input_sum = tl.sum(sum_input, 1)[:, None]
    
    tl.store(output_grad_ptr + (x3), output_grad_sum, xmask)
    tl.store(output_ptr + (x3), output_input_sum, xmask)