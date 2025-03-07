# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_1(input_ptr_mean, input_ptr_var, input_ptr_inv_std, input_ptr_gamma, input_ptr_beta, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    linear_index = block_indices
    batch_index = linear_index // 4356
    channel_index = (linear_index // 4356) % 64
    
    mean = tl.load(input_ptr_mean + (linear_index), mask)
    var = tl.load(input_ptr_var + (batch_index // 8), mask, eviction_policy='evict_last')
    inv_std = tl.load(input_ptr_inv_std + (batch_index // 8), mask, eviction_policy='evict_last')
    gamma = tl.load(input_ptr_gamma + (channel_index), mask, eviction_policy='evict_last')
    beta = tl.load(input_ptr_beta + (channel_index), mask, eviction_policy='evict_last')
    
    half = 0.5
    scaled_mean = mean * half
    sqrt_inv_two = 0.7071067811865476
    erf_input = mean * sqrt_inv_two
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    one = 1.0
    erf_adjusted = erf_result + one
    gelu_output = scaled_mean * erf_adjusted
    centered_output = gelu_output - var
    
    eps = 34848.0
    var_adjusted = var / eps
    epsilon = 1e-05
    var_with_epsilon = var_adjusted + epsilon
    inv_std_adjusted = tl.extra.cuda.libdevice.rsqrt(var_with_epsilon)
    normalized_output = centered_output * inv_std_adjusted
    scaled_output = normalized_output * gamma
    final_output = scaled_output + beta
    
    tl.store(output_ptr + (linear_index), final_output, mask)