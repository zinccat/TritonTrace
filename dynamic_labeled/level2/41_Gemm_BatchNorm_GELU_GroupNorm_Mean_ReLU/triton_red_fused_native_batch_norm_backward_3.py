# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_batch_norm_backward_3(
    input_grad_ptr, mean_ptr, variance_ptr, input_ptr, output_grad_mean_ptr, output_grad_var_ptr, output_grad_input_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x_indices = xindex
    temp_sum_mean = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_grad = tl.load(input_grad_ptr + (x_indices), xmask, eviction_policy='evict_last')
    temp_sum_var = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r_indices = rindex
        grad_input = tl.load(input_grad_ptr + (x_indices + 1024 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        grad_output = tl.load(mean_ptr + (x_indices + 1024 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        
        broadcast_grad_input = tl.broadcast_to(grad_input, [XBLOCK, RBLOCK])
        temp_sum_mean += broadcast_grad_input
        temp_sum_mean = tl.where(rmask & xmask, temp_sum_mean, temp_sum_mean)
        
        grad_diff = grad_output - input_grad
        grad_input_diff = grad_input * grad_diff
        broadcast_grad_input_diff = tl.broadcast_to(grad_input_diff, [XBLOCK, RBLOCK])
        temp_sum_var += broadcast_grad_input_diff
        temp_sum_var = tl.where(rmask & xmask, temp_sum_var, temp_sum_var)
    
    sum_mean = tl.sum(temp_sum_mean, 1)[:, None]
    sum_var = tl.sum(temp_sum_var, 1)[:, None]
    
    tl.store(output_grad_mean_ptr + (x_indices), sum_mean, xmask)
    tl.store(output_grad_var_ptr + (x_indices), sum_var, xmask)
    
    input = tl.load(input_ptr + (x_indices), xmask, eviction_policy='evict_last')
    output_grad_input = sum_var * input
    tl.store(output_grad_input_ptr + (x_indices), output_grad_input, xmask)