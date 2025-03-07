# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_batch_norm_backward_3(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, output_grad_ptr, output_mean_ptr, output_var_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    temp_sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum_grad_squared = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    mean_values = tl.load(mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        input_values = tl.load(input_ptr + (x0 + 1024 * r1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        grad_values = tl.load(input_grad_ptr + (x0 + 1024 * r1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        
        broadcast_input = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
        temp_sum_grad += broadcast_input
        temp_sum_grad = tl.where(r_mask & x_mask, temp_sum_grad, temp_sum_grad)
        
        grad_diff = grad_values - mean_values
        grad_diff_scaled = input_values * grad_diff
        broadcast_grad_diff_scaled = tl.broadcast_to(grad_diff_scaled, [XBLOCK, RBLOCK])
        temp_sum_grad_squared += broadcast_grad_diff_scaled
        temp_sum_grad_squared = tl.where(r_mask & x_mask, temp_sum_grad_squared, temp_sum_grad_squared)
    
    output_mean = tl.sum(temp_sum_grad, 1)[:, None]
    output_var = tl.sum(temp_sum_grad_squared, 1)[:, None]
    
    tl.store(output_mean_ptr + (x0), output_mean, x_mask)
    tl.store(output_var_ptr + (x0), output_var, x_mask)
    
    variance_values = tl.load(variance_ptr + (x0), x_mask, eviction_policy='evict_last')
    output_var_scaled = output_var * variance_values
    
    tl.store(output_grad_ptr + (x0), output_var_scaled, x_mask)