# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_49poi_fused__native_batch_norm_legit_functional_add_relu_49(
    in_out_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, beta_ptr, input_data_ptr, running_mean_ptr, running_var_ptr, epsilon_ptr, scale_ptr, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = x_index
    x0 = (x_index % 2048)
    
    input_data = tl.load(input_data_ptr + (x2), None)
    mean = tl.load(mean_ptr + (x0), None, eviction_policy='evict_last')
    variance = tl.load(variance_ptr + (x0), None, eviction_policy='evict_last')
    gamma = tl.load(gamma_ptr + (x0), None, eviction_policy='evict_last')
    beta = tl.load(beta_ptr + (x0), None, eviction_policy='evict_last')
    running_mean = tl.load(input_data_ptr + (x2), None)
    running_var = tl.load(running_mean_ptr + (x0), None, eviction_policy='evict_last')
    epsilon = tl.load(running_var_ptr + (x0), None, eviction_policy='evict_last')
    scale = tl.load(epsilon_ptr + (x0), None, eviction_policy='evict_last')
    
    normalized_input = input_data - mean
    variance_scale = 490.0
    normalized_variance = variance / variance_scale
    epsilon_value = 1e-05
    adjusted_variance = normalized_variance + epsilon_value
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_input = normalized_input * inv_sqrt_variance
    scaled_gamma = scaled_input * gamma
    batch_norm_output = scaled_gamma + beta
    
    running_mean_diff = running_mean - running_var
    running_var_scale = variance / variance_scale
    adjusted_running_var = running_var_scale + epsilon_value
    inv_sqrt_running_var = tl.extra.cuda.libdevice.rsqrt(adjusted_running_var)
    scaled_running_mean = running_mean_diff * inv_sqrt_running_var
    scaled_scale = scaled_running_mean * scale
    final_output = scaled_scale + epsilon
    
    relu_output = batch_norm_output + final_output
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_result = triton_helpers.maximum(zero_tensor, relu_output)
    
    tl.store(in_out_ptr + (x2), relu_result, None)