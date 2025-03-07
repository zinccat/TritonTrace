# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_3poi_fused_native_group_norm_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x4 = x_index
    x6 = x_index // 16384
    x7 = ((x_index // 1024) % 64)
    
    input_data = tl.load(in_ptr0 + (x4), None)
    weight_data = tl.load(in_ptr1 + (x6), None, eviction_policy='evict_last')
    bias_data = tl.load(in_ptr2 + (x7), None, eviction_policy='evict_last')
    
    grad_output = tl.load(in_out_ptr0 + (x4), None)
    running_mean = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    running_var = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    saved_mean = tl.load(in_ptr5 + (x6), None, eviction_policy='evict_last')
    
    weight_bias_product = weight_data * bias_data
    input_weight_bias_product = input_data * weight_bias_product
    
    running_mean_var_product = running_mean * running_var
    var_diff = running_mean_var_product - saved_mean
    
    var_diff_weight = var_diff * weight_data
    var_diff_weight_cubed = var_diff_weight * var_diff_weight * var_diff_weight
    
    epsilon = 6.103515625e-05
    normalized_grad = var_diff_weight_cubed * epsilon
    
    grad_input = grad_output * normalized_grad
    adjusted_input = input_weight_bias_product + grad_input
    
    bias_adjustment = -normalized_grad * running_var
    mean_adjustment = running_mean * var_diff_weight
    mean_adjustment_scaled = mean_adjustment * epsilon
    
    final_adjustment = bias_adjustment - mean_adjustment_scaled
    adjusted_output = adjusted_input + final_adjustment
    
    tl.store(in_out_ptr0 + (x4), adjusted_output, None)