# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_5poi_fused_native_group_norm_5(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    block_index = index // 1024
    channel_index = (block_index % 64)
    
    mean_value = tl.load(input_ptr_mean + (index), None)
    var_value = tl.load(input_ptr_var + (block_index // 16), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (block_index // 16), None, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (channel_index), None, eviction_policy='evict_last')
    input_value = tl.load(input_ptr_input + (channel_index), None, eviction_policy='evict_last')
    
    normalized_value = input_value - mean_value
    variance_scaled = 16384.0
    normalized_variance = var_value / variance_scaled
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    
    scaled_normalized_value = normalized_value * inv_sqrt_variance
    scaled_and_shifted_value = scaled_normalized_value * scale_value
    output_value = scaled_and_shifted_value + bias_value
    
    tl.store(output_ptr + (index), output_value, None)