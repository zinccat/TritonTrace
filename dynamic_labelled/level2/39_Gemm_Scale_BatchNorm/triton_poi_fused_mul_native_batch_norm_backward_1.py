# From: 39_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_native_batch_norm_backward_1(
    input_grad_ptr, scale_ptr, running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr, grad_output_ptr, grad_input_ptr, 
    output_grad_ptr, output_input_grad_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < num_elements
    x2 = xindex
    x0 = (xindex % 512)
    
    input_grad = tl.load(input_grad_ptr + (x2), xmask)
    scale = tl.load(scale_ptr + (x2), xmask)
    running_mean = tl.load(running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')
    bn_weight = tl.load(bn_weight_ptr + (x0), xmask, eviction_policy='evict_last')
    bn_weight_squared = tl.load(bn_bias_ptr + (x0), xmask, eviction_policy='evict_last')
    bn_bias = tl.load(grad_output_ptr + (x0), xmask, eviction_policy='evict_last')
    grad_output = tl.load(grad_input_ptr + (x0), xmask, eviction_policy='evict_last')
    
    scale_running_mean = scale * running_mean
    mean_diff = scale_running_mean - running_var
    normalization_factor = (tl.full([], 1.0, tl.float64) / ((512 * kernel_size) / 512))
    normalization_factor_float32 = normalization_factor.to(tl.float32)
    normalized_bn_weight = bn_weight * normalization_factor_float32
    variance_term = bn_weight_squared * bn_weight_squared
    weight_variance_product = normalized_bn_weight * variance_term
    adjusted_mean_diff = mean_diff * weight_variance_product
    adjusted_input_grad = input_grad - adjusted_mean_diff
    adjusted_bn_bias = bn_bias * normalization_factor_float32
    final_input_grad = adjusted_input_grad - adjusted_bn_bias
    grad_output_weight_product = bn_weight_squared * grad_output
    final_output_grad = final_input_grad * grad_output_weight_product
    output_grad_with_scale = final_output_grad * running_mean
    
    tl.store(output_grad_ptr + (x2), final_output_grad, xmask)
    tl.store(output_input_grad_ptr + (x2), output_grad_with_scale, xmask)