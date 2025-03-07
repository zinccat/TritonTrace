# From: 37_Matmul_Swish_Sum_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_0(
    input_grad_ptr, input_ptr, group_norm_mean_ptr, group_norm_var_ptr, input_data_ptr, group_norm_weight_ptr, 
    output_grad_ptr, output_data_ptr, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_block = rindex
    x_block = xindex
    x_modulo = xindex % 32
    grad_input = tl.load(input_grad_ptr + (r_block + 32 * x_block), xmask, other=0.0)
    input_data = tl.load(input_ptr + (r_block + 32 * x_block), xmask, other=0.0)
    group_norm_mean = tl.load(group_norm_mean_ptr + (r_block + 32 * x_modulo), xmask, eviction_policy='evict_last', other=0.0)
    group_norm_var = tl.load(group_norm_var_ptr + (r_block + 32 * x_modulo), xmask, eviction_policy='evict_last', other=0.0)
    input_data_full = tl.load(input_data_ptr + (x_block), xmask, eviction_policy='evict_last')
    group_norm_weight_full = tl.load(group_norm_weight_ptr + (x_block), xmask, eviction_policy='evict_last')
    
    sigmoid_input = tl.sigmoid(input_data)
    sigmoid_output = sigmoid_input * input_data
    group_norm_sum = sigmoid_output + group_norm_mean
    grad_input_scaled = grad_input * group_norm_sum
    grad_input_var_scaled = grad_input_scaled * group_norm_var
    grad_input_var_broadcast = tl.broadcast_to(grad_input_var_scaled, [XBLOCK, RBLOCK])
    grad_input_var_masked = tl.where(xmask, grad_input_var_broadcast, 0)
    grad_input_var_sum = tl.sum(grad_input_var_masked, 1)[:, None]
    
    grad_input_group_norm_var = grad_input * group_norm_var
    grad_input_group_norm_var_broadcast = tl.broadcast_to(grad_input_group_norm_var, [XBLOCK, RBLOCK])
    grad_input_group_norm_var_masked = tl.where(xmask, grad_input_group_norm_var_broadcast, 0)
    grad_input_group_norm_var_sum = tl.sum(grad_input_group_norm_var_masked, 1)[:, None]
    
    input_data_group_norm_var = input_data_full * group_norm_var
    grad_input_input_data_group_norm_var = grad_input * input_data_group_norm_var
    group_norm_var_weight_sum = grad_input_var_sum * group_norm_weight_full
    group_norm_var_weight_diff = group_norm_var_weight_sum - grad_input_var_sum
    group_norm_var_weight_scaled = group_norm_var_weight_diff * input_data_full
    group_norm_var_weight_cubed = group_norm_var_weight_scaled * input_data_full
    group_norm_var_weight_cubed_scaled = group_norm_var_weight_cubed * input_data_full
    scaling_factor = 0.03125
    group_norm_var_weight_scaled_factor = group_norm_var_weight_cubed_scaled * scaling_factor
    group_norm_sum_scaled_factor = group_norm_sum * group_norm_var_weight_scaled_factor
    grad_input_input_data_group_norm_var_scaled = grad_input_input_data_group_norm_var + group_norm_sum_scaled_factor
    
    group_norm_var_weight_diff_scaled = -group_norm_var_weight_scaled_factor
    group_norm_var_weight_diff_scaled_factor = group_norm_var_weight_diff_scaled * group_norm_weight_full
    group_norm_var_weight_input_data = grad_input_var_sum * input_data_full
    group_norm_var_weight_input_data_scaled = group_norm_var_weight_input_data * scaling_factor
    group_norm_var_weight_diff_scaled_final = group_norm_var_weight_diff_scaled_factor - group_norm_var_weight_input_data_scaled
    grad_input_input_data_group_norm_var_final = grad_input_input_data_group_norm_var_scaled + group_norm_var_weight_diff_scaled_final
    
    grad_output_scaled = grad_input_input_data_group_norm_var_final * sigmoid_input
    grad_output_scaled_input_data = grad_input_input_data_group_norm_var_final * input_data
    sigmoid_output_complement = 1.0 - sigmoid_input
    sigmoid_output_scaled_input_data = grad_output_scaled_input_data * (sigmoid_input * sigmoid_output_complement)
    grad_output_final = grad_output_scaled + sigmoid_output_scaled_input_data
    
    tl.store(output_grad_ptr + (r_block + 32 * x_block), grad_input_input_data_group_norm_var_final, xmask)
    tl.store(output_data_ptr + (r_block + 32 * x_block), grad_output_final, xmask)