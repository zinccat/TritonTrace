# From: 37_Matmul_Swish_Sum_GroupNorm

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_native_group_norm_1(input_ptr_mean, input_ptr_var, input_ptr_mean_buffer, input_ptr_var_buffer, input_ptr_gamma, input_ptr_beta, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    element_indices = block_indices
    element_indices_mod = block_indices % 1024
    
    mean = tl.load(input_ptr_mean + (element_indices), None)
    var = tl.load(input_ptr_var + (element_indices_mod), None, eviction_policy='evict_last')
    mean_buffer = tl.load(input_ptr_mean_buffer + (element_indices // 32), None, eviction_policy='evict_last')
    var_buffer = tl.load(input_ptr_var_buffer + (element_indices // 32), None, eviction_policy='evict_last')
    gamma = tl.load(input_ptr_gamma + (element_indices_mod), None, eviction_policy='evict_last')
    beta = tl.load(input_ptr_beta + (element_indices_mod), None, eviction_policy='evict_last')
    
    swish = tl.sigmoid(mean)
    swish_scaled = swish * mean
    normalized_mean = swish_scaled + var
    centered_mean = normalized_mean - mean_buffer
    
    variance_scale = 32.0
    epsilon = 1e-05
    adjusted_variance = var_buffer / variance_scale
    variance_with_epsilon = adjusted_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    
    normalized_output = centered_mean * inv_sqrt_variance
    scaled_output = normalized_output * gamma
    final_output = scaled_output + beta
    
    tl.store(output_ptr + (element_indices), final_output, None)