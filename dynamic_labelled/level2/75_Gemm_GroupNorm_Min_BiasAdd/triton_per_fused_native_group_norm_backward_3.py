# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_backward_3(
    input_grad_ptr, input_ptr, running_mean_ptr, running_var_ptr, input_data_ptr, output_grad_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_index = rindex
    col_index = xindex
    col_mod_index = xindex % 8
    grad_input = tl.load(input_grad_ptr + (row_index + 32 * col_index), xmask, other=0.0)
    grad_output = tl.load(input_ptr + (row_index + 32 * col_index), xmask, other=0.0)
    running_var = tl.load(running_var_ptr + (row_index + 32 * col_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    running_mean = tl.load(running_mean_ptr + (col_index), xmask, eviction_policy='evict_last')
    input_data = tl.load(input_data_ptr + (col_index), xmask, eviction_policy='evict_last')
    
    grad_input_output_product = grad_input * grad_output
    grad_input_var_product = grad_input_output_product * running_var
    broadcast_grad_input_var_product = tl.broadcast_to(grad_input_var_product, [XBLOCK, RBLOCK])
    masked_grad_input_var_product = tl.where(xmask, broadcast_grad_input_var_product, 0)
    sum_grad_input_var_product = tl.sum(masked_grad_input_var_product, 1)[:, None]
    
    grad_input_var = grad_input * running_var
    broadcast_grad_input_var = tl.broadcast_to(grad_input_var, [XBLOCK, RBLOCK])
    masked_grad_input_var = tl.where(xmask, broadcast_grad_input_var, 0)
    sum_grad_input_var = tl.sum(masked_grad_input_var, 1)[:, None]
    
    mean_var_product = running_mean * running_var
    grad_input_mean_var_product = grad_input * mean_var_product
    sum_grad_input_var_mean = sum_grad_input_var * running_mean
    sum_grad_input_var_mean_diff = sum_grad_input_var_mean - sum_grad_input_var
    mean_var_product_scaled = sum_grad_input_var_mean_diff * running_mean
    mean_var_product_scaled_cubed = mean_var_product_scaled * running_mean * running_mean
    scale_factor = 0.03125
    scaled_mean_var_product = mean_var_product_scaled_cubed * scale_factor
    grad_output_scaled_mean_var = grad_output * scaled_mean_var_product
    grad_input_mean_var_sum = grad_input_mean_var_product + grad_output_scaled_mean_var
    
    neg_scaled_mean_var = -scaled_mean_var
    neg_scaled_mean_var_output = neg_scaled_mean_var * running_mean
    mean_var_sum = sum_grad_input_var * running_mean
    mean_var_sum_scaled = mean_var_sum * scale_factor
    neg_scaled_mean_var_output_diff = neg_scaled_mean_var_output - mean_var_sum_scaled
    final_output = grad_input_mean_var_sum + neg_scaled_mean_var_output_diff
    
    tl.store(output_grad_ptr + (row_index + 32 * col_index), final_output, xmask)