# From: 37_Matmul_Swish_Sum_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_0(
    input_grad_ptr, input_ptr, weight_grad_ptr, weight_ptr, group_norm_mean_ptr, group_norm_var_ptr, 
    output_grad_ptr, output_ptr, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_index = rindex
    col_index = xindex
    col_modulo = xindex % 32
    input_grad = tl.load(input_grad_ptr + (row_index + 32 * col_index), xmask, other=0.0)
    input = tl.load(input_ptr + (row_index + 32 * col_index), xmask, other=0.0)
    group_norm_mean = tl.load(group_norm_mean_ptr + (row_index + 32 * col_modulo), xmask, eviction_policy='evict_last', other=0.0)
    group_norm_var = tl.load(group_norm_var_ptr + (row_index + 32 * col_modulo), xmask, eviction_policy='evict_last', other=0.0)
    group_norm_mean_broadcast = tl.load(group_norm_mean_ptr + (col_index), xmask, eviction_policy='evict_last')
    group_norm_var_broadcast = tl.load(group_norm_var_ptr + (col_index), xmask, eviction_policy='evict_last')
    
    sigmoid_input = tl.sigmoid(input)
    sigmoid_input_grad = sigmoid_input * input
    group_norm_grad = sigmoid_input_grad + group_norm_mean
    input_grad_weight = input_grad * group_norm_grad
    weight_grad_input = input_grad_weight * group_norm_var
    weight_grad_input_broadcast = tl.broadcast_to(weight_grad_input, [XBLOCK, RBLOCK])
    masked_weight_grad_input = tl.where(xmask, weight_grad_input_broadcast, 0)
    sum_weight_grad_input = tl.sum(masked_weight_grad_input, 1)[:, None]
    
    input_grad_var = input_grad * group_norm_var
    input_grad_var_broadcast = tl.broadcast_to(input_grad_var, [XBLOCK, RBLOCK])
    masked_input_grad_var = tl.where(xmask, input_grad_var_broadcast, 0)
    sum_input_grad_var = tl.sum(masked_input_grad_var, 1)[:, None]
    
    group_norm_mean_grad = group_norm_mean_broadcast * group_norm_var
    input_grad_group_norm_mean = input_grad * group_norm_mean_grad
    sum_group_norm_var_grad = sum_input_grad_var * group_norm_var_broadcast
    group_norm_mean_grad_adjusted = sum_group_norm_var_grad - sum_weight_grad_input
    group_norm_mean_grad_scaled = group_norm_mean_grad_adjusted * group_norm_mean_broadcast
    group_norm_mean_grad_scaled_cubed = group_norm_mean_grad_scaled * group_norm_mean_grad_scaled * group_norm_mean_broadcast
    scaling_factor = 0.03125
    group_norm_mean_grad_scaled_cubed_scaled = group_norm_mean_grad_scaled_cubed * scaling_factor
    group_norm_grad_adjusted = group_norm_grad * group_norm_mean_grad_scaled_cubed_scaled
    output_grad_group_norm = input_grad_group_norm_mean + group_norm_grad_adjusted
    
    group_norm_var_grad_adjusted = -group_norm_mean_grad_scaled_cubed_scaled * group_norm_var_broadcast
    group_norm_var_grad_adjusted_scaled = sum_group_norm_var_grad * group_norm_mean_broadcast
    group_norm_var_grad_adjusted_scaled_scaled = group_norm_var_grad_adjusted_scaled * scaling_factor
    group_norm_var_grad_final = group_norm_var_grad_adjusted + group_norm_var_grad_adjusted_scaled_scaled
    output_grad_group_norm_var = output_grad_group_norm + group_norm_var_grad_final
    
    output_grad_sigmoid = output_grad_group_norm * sigmoid_input
    output_grad_input = output_grad_group_norm * input
    sigmoid_grad = sigmoid_input * (1.0 - sigmoid_input)
    output_grad_input_scaled = output_grad_input * sigmoid_grad
    output_grad_final = output_grad_sigmoid + output_grad_input_scaled
    
    tl.store(output_grad_ptr + (row_index + 32 * col_index), output_grad_group_norm, xmask)
    tl.store(output_ptr + (row_index + 32 * col_index), output_grad_final, xmask)