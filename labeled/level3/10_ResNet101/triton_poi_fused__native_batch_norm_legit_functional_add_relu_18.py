# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_18poi_fused__native_batch_norm_legit_functional_add_relu_18(
    in_out_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, beta_ptr, input_data_ptr, running_mean_ptr, running_var_ptr, epsilon_ptr, scale_ptr, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    index2 = index
    index0 = (index % 256)
    
    input_data = tl.load(input_ptr + (index2), None)
    mean = tl.load(mean_ptr + (index0), None, eviction_policy='evict_last')
    variance = tl.load(variance_ptr + (index0), None, eviction_policy='evict_last')
    gamma = tl.load(gamma_ptr + (index0), None, eviction_policy='evict_last')
    beta = tl.load(beta_ptr + (index0), None, eviction_policy='evict_last')
    input_data2 = tl.load(input_data_ptr + (index2), None)
    running_mean = tl.load(running_mean_ptr + (index0), None, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (index0), None, eviction_policy='evict_last')
    epsilon = tl.load(epsilon_ptr + (index0), None, eviction_policy='evict_last')
    scale = tl.load(scale_ptr + (index0), None, eviction_policy='evict_last')
    
    normalized_input = input_data - mean
    variance_scale = 31360.0
    normalized_variance = variance / variance_scale
    epsilon_value = 1e-05
    adjusted_variance = normalized_variance + epsilon_value
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_input = normalized_input * inv_sqrt_variance
    scaled_gamma = scaled_input * gamma
    batch_norm_output = scaled_gamma + beta
    
    normalized_input2 = input_data2 - running_mean
    normalized_running_var = running_var / variance_scale
    adjusted_running_var = normalized_running_var + epsilon_value
    inv_sqrt_running_var = tl.extra.cuda.libdevice.rsqrt(adjusted_running_var)
    scaled_input2 = normalized_input2 * inv_sqrt_running_var
    scaled_scale = scaled_input2 * scale
    final_output = scaled_scale + running_var
    
    fused_output = batch_norm_output + final_output
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, fused_output)
    
    tl.store(in_out_ptr + (index2), relu_output, None)