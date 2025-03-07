# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_3(
    input_grad_ptr, input_data_ptr, input_mean_ptr, input_var_ptr, input_scale_ptr,
    output_grad_ptr, output_mean_ptr, output_var_ptr, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x_indices = xindex
    temp_sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum_var = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    input_var_values = tl.load(input_var_ptr + (x_indices), xmask, eviction_policy='evict_last')
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r_indices = rindex
        grad_values = tl.load(input_grad_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        data_values = tl.load(input_data_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean_values = tl.load(input_mean_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        
        sigmoid_data = tl.sigmoid(data_values)
        grad_scaled = grad_values * sigmoid_data
        grad_unscaled = grad_values * data_values
        one = 1.0
        one_minus_sigmoid = one - sigmoid_data
        grad_adjusted = sigmoid_data * one_minus_sigmoid
        grad_var = grad_unscaled * grad_adjusted
        grad_combined = grad_scaled + grad_var
        grad_weighted = grad_combined * one
        grad_broadcast = tl.broadcast_to(grad_weighted, [XBLOCK, RBLOCK])
        temp_sum_grad += grad_broadcast
        temp_sum_grad = tl.where(rmask & xmask, temp_sum_grad, temp_sum_grad)
        
        mean_adjusted = mean_values - input_var_values
        var_weighted = grad_weighted * mean_adjusted
        var_broadcast = tl.broadcast_to(var_weighted, [XBLOCK, RBLOCK])
        temp_sum_var += var_broadcast
        temp_sum_var = tl.where(rmask & xmask, temp_sum_var, temp_sum_var)
    
    output_grad_sum = tl.sum(temp_sum_grad, 1)[:, None]
    output_var_sum = tl.sum(temp_sum_var, 1)[:, None]
    
    tl.store(output_grad_ptr + (x_indices), output_grad_sum, xmask)
    tl.store(output_mean_ptr + (x_indices), output_var_sum, xmask)
    
    scale_values = tl.load(input_scale_ptr + (x_indices), xmask, eviction_policy='evict_last')
    output_var_scaled = output_var_sum * scale_values
    tl.store(output_var_ptr + (x_indices), output_var_scaled, xmask)