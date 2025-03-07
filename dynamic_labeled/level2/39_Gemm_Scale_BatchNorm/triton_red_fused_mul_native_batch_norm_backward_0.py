# From: 39_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_native_batch_norm_backward_0(
    input_grad_ptr, input_data_ptr, mean_ptr, variance_ptr, scale_ptr, 
    output_grad_ptr, output_data_ptr, output_scaled_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x_indices = xindex
    zero_matrix = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    mean_values = tl.load(mean_ptr + (x_indices), xmask, eviction_policy='evict_last')
    variance_values = tl.load(variance_ptr + (x_indices), xmask, eviction_policy='evict_last')
    accumulated_grad = zero_matrix
    accumulated_scaled_grad = zero_matrix

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r_indices = rindex
        grad_values = tl.load(input_grad_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        data_values = tl.load(input_data_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        
        broadcast_grad = tl.broadcast_to(grad_values, [XBLOCK, RBLOCK])
        accumulated_grad += broadcast_grad
        accumulated_grad = tl.where(rmask & xmask, accumulated_grad, zero_matrix)
        
        scaled_diff = (data_values * mean_values) - variance_values
        scaled_grad = grad_values * scaled_diff
        broadcast_scaled_grad = tl.broadcast_to(scaled_grad, [XBLOCK, RBLOCK])
        accumulated_scaled_grad += broadcast_scaled_grad
        accumulated_scaled_grad = tl.where(rmask & xmask, accumulated_scaled_grad, zero_matrix)

    summed_grad = tl.sum(accumulated_grad, 1)[:, None]
    summed_scaled_grad = tl.sum(accumulated_scaled_grad, 1)[:, None]
    
    tl.store(output_grad_ptr + (x_indices), summed_grad, xmask)
    tl.store(output_data_ptr + (x_indices), summed_scaled_grad, xmask)
    
    scale_values = tl.load(scale_ptr + (x_indices), xmask, eviction_policy='evict_last')
    scaled_output = summed_scaled_grad * scale_values
    tl.store(output_scaled_ptr + (x_indices), scaled_output, xmask)