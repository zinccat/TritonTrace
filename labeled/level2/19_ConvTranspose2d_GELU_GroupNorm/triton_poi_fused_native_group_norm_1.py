# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_native_group_norm_1(input_ptr_mean, input_ptr_var, input_ptr_inv_std, input_ptr_gamma, input_ptr_beta, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    batch_index = index // 4356
    channel_index = (index // 4356) % 64
    
    input_mean = tl.load(input_ptr_mean + (index), None)
    input_var = tl.load(input_ptr_var + (batch_index // 8), None, eviction_policy='evict_last')
    input_inv_std = tl.load(input_ptr_inv_std + (batch_index // 8), None, eviction_policy='evict_last')
    input_gamma = tl.load(input_ptr_gamma + (channel_index), None, eviction_policy='evict_last')
    input_beta = tl.load(input_ptr_beta + (channel_index), None, eviction_policy='evict_last')
    
    half = 0.5
    scaled_input = input_mean * half
    sqrt_half = 0.7071067811865476
    erf_input = input_mean * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    one = 1.0
    erf_scaled = erf_result + one
    normalized_input = scaled_input * erf_scaled
    mean_centered = normalized_input - input_var
    
    var_scale = 34848.0
    normalized_var = input_inv_std / var_scale
    epsilon = 1e-05
    adjusted_var = normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)
    scaled_output = mean_centered * inv_std
    gamma_scaled = scaled_output * input_gamma
    output = gamma_scaled + input_beta
    
    tl.store(output_ptr + (index), output, None)