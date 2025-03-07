# From: 39_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_native_batch_norm_backward_0(
    input_grad_ptr, input_data_ptr, mean_ptr, variance_ptr, scale_ptr, 
    output_grad_ptr, output_mean_ptr, output_var_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x_indices = xindex
    temp_sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    mean_values = tl.load(mean_ptr + (x_indices), xmask, eviction_policy='evict_last')
    variance_values = tl.load(variance_ptr + (x_indices), xmask, eviction_policy='evict_last')
    temp_sum_weighted_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r_indices = rindex
        grad_values = tl.load(input_grad_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        data_values = tl.load(input_data_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        
        broadcast_grad = tl.broadcast_to(grad_values, [XBLOCK, RBLOCK])
        temp_sum_grad += broadcast_grad
        temp_sum_grad = tl.where(rmask & xmask, temp_sum_grad, temp_sum_grad)
        
        grad_variance = data_values * (mean_values - variance_values)
        weighted_grad = grad_values * grad_variance
        broadcast_weighted_grad = tl.broadcast_to(weighted_grad, [XBLOCK, RBLOCK])
        temp_sum_weighted_grad += broadcast_weighted_grad
        temp_sum_weighted_grad = tl.where(rmask & xmask, temp_sum_weighted_grad, temp_sum_weighted_grad)

    sum_grad = tl.sum(temp_sum_grad, 1)[:, None]
    sum_weighted_grad = tl.sum(temp_sum_weighted_grad, 1)[:, None]
    
    tl.store(output_grad_ptr + (x_indices), sum_grad, xmask)
    tl.store(output_mean_ptr + (x_indices), sum_weighted_grad, xmask)
    
    scale_values = tl.load(scale_ptr + (x_indices), xmask, eviction_policy='evict_last')
    output_var = sum_weighted_grad * scale_values
    tl.store(output_var_ptr + (x_indices), output_var, xmask)