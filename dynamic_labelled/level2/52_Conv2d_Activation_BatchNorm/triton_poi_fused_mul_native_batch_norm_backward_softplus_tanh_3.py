# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_native_batch_norm_backward_softplus_tanh_3poi_fused_mul_native_batch_norm_backward_softplus_tanh_3(
    input_grad_ptr, input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, saved_mean_ptr, saved_inv_std_ptr,
    output_grad_ptr, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size_0) % 16)
    
    input_grad = tl.load(input_grad_ptr + (x3), x_mask, eviction_policy='evict_last')
    input = tl.load(input_ptr + (x3), x_mask, eviction_policy='evict_last')
    running_mean = tl.load(running_mean_ptr + (x1), x_mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x1), x_mask, eviction_policy='evict_last')
    weight = tl.load(weight_ptr + (x1), x_mask, eviction_policy='evict_last')
    bias = tl.load(bias_ptr + (x1), x_mask, eviction_policy='evict_last')
    saved_mean = tl.load(saved_mean_ptr + (x1), x_mask, eviction_policy='evict_last')
    saved_inv_std = tl.load(saved_inv_std_ptr + (x1), x_mask, eviction_policy='evict_last')
    
    threshold = 20.0
    is_large = input > threshold
    exp_input = tl.math.exp(input)
    log1p_exp_input = tl.extra.cuda.libdevice.log1p(exp_input)
    softplus_input = tl.where(is_large, input, log1p_exp_input)
    tanh_softplus = tl.extra.cuda.libdevice.tanh(softplus_input)
    softplus_tanh = tanh_softplus * input
    
    normalized_diff = softplus_tanh - running_mean
    normalization_factor = (tl.full([], 1.0, tl.float64) / ((64 * kernel_size_1 + ((-64) * kernel_size_1 * kernel_size_2) + 16 * kernel_size_1 * kernel_size_2 * kernel_size_2) / 16))
    normalization_factor = normalization_factor.to(tl.float32)
    weight_normalized = weight * normalization_factor
    var_squared = running_var * running_var
    weight_var_squared = weight_normalized * var_squared
    scaled_diff = normalized_diff * weight_var_squared
    input_grad_adjusted = input_grad - scaled_diff
    
    bias_normalized = bias * normalization_factor
    input_grad_adjusted -= bias_normalized
    weight_saved_inv_std = weight * saved_inv_std
    final_output_grad = input_grad_adjusted * weight_saved_inv_std
    
    tl.store(output_grad_ptr + (x3), final_output_grad, x_mask)