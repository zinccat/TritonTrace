# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_batch_norm_backward_tanh_backward_8(
    in_out_ptr, input_data, grad_output, running_mean, running_var, weight, bias, scale, kernel_size, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    
    grad_input = tl.load(in_out_ptr + (x3), None)
    input_data_val = tl.load(input_data + (x3), None)
    grad_output_val = tl.load(grad_output + (x3), None)
    running_mean_val = tl.load(running_mean + (x1), None, eviction_policy='evict_last')
    running_var_val = tl.load(running_var + (x1), None, eviction_policy='evict_last')
    weight_val = tl.load(weight + (x1), None, eviction_policy='evict_last')
    scale_val = tl.load(scale + (x1), None, eviction_policy='evict_last')
    bias_val = tl.load(bias + (x1), None, eviction_policy='evict_last')
    
    input_data_squared = input_data_val * input_data_val
    variance_epsilon = 1.0
    normalized_variance = variance_epsilon - input_data_squared
    normalized_grad_input = grad_input * normalized_variance
    
    mean_diff = grad_output_val - running_mean_val
    normalization_factor = (tl.full([], 1.00000000000000, tl.float64) / ((262144 * kernel_size) / 64))
    normalization_factor_float32 = normalization_factor.to(tl.float32)
    normalized_weight = weight_val * normalization_factor_float32
    variance_squared = running_var_val * running_var_val
    weight_variance_product = normalized_weight * variance_squared
    mean_variance_product = mean_diff * weight_variance_product
    adjusted_grad_input = normalized_grad_input - mean_variance_product
    
    normalized_scale = scale_val * normalization_factor_float32
    adjusted_grad_input_with_scale = adjusted_grad_input - normalized_scale
    
    bias_weight_product = weight_val * bias_val
    final_grad_input = adjusted_grad_input_with_scale * bias_weight_product
    
    tl.store(in_out_ptr + (x3), final_grad_input, None)