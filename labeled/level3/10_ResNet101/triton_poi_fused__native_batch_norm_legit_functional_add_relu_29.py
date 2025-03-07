# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_29poi_fused__native_batch_norm_legit_functional_add_relu_29(
    in_out_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, beta_ptr, input_data_ptr, running_mean_ptr, running_var_ptr, epsilon_ptr, scale_ptr, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    element_index = index
    block_index = (index % 512)
    
    input_data = tl.load(input_ptr + (element_index), None)
    mean = tl.load(mean_ptr + (block_index), None, eviction_policy='evict_last')
    variance = tl.load(variance_ptr + (block_index), None, eviction_policy='evict_last')
    gamma = tl.load(gamma_ptr + (block_index), None, eviction_policy='evict_last')
    beta = tl.load(beta_ptr + (block_index), None, eviction_policy='evict_last')
    input_data_2 = tl.load(input_data_ptr + (element_index), None)
    running_mean = tl.load(running_mean_ptr + (block_index), None, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (block_index), None, eviction_policy='evict_last')
    epsilon = tl.load(epsilon_ptr + (block_index), None, eviction_policy='evict_last')
    scale = tl.load(scale_ptr + (block_index), None, eviction_policy='evict_last')
    
    normalized_input = input_data - mean
    variance_scale = 7840.0
    variance_adjusted = variance / variance_scale
    epsilon_value = 1e-05
    variance_adjusted_epsilon = variance_adjusted + epsilon_value
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)
    normalized_input_scaled = normalized_input * inv_sqrt_variance
    scaled_input = normalized_input_scaled * gamma
    batch_norm_output = scaled_input + beta
    
    input_data_2_normalized = input_data_2 - running_mean
    running_var_adjusted = running_var / variance_scale
    running_var_adjusted_epsilon = running_var_adjusted + epsilon_value
    inv_sqrt_running_var = tl.extra.cuda.libdevice.rsqrt(running_var_adjusted_epsilon)
    running_var_scaled_input = input_data_2_normalized * inv_sqrt_running_var
    scaled_running_var = running_var_scaled_input * scale
    final_output = scaled_running_var + running_var
    
    fused_output = batch_norm_output + final_output
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, fused_output)
    
    tl.store(in_out_ptr + (element_index), relu_output, None)