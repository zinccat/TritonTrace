# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_native_group_norm_5(input_ptr_mean, input_ptr_var, input_ptr_inv_std, input_ptr_weight, input_ptr_bias, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    block_index = index // 1024
    channel_index = block_index % 64
    
    mean_value = tl.load(input_ptr_mean + (index), None)
    var_value = tl.load(input_ptr_var + (block_index // 16), None, eviction_policy='evict_last')
    inv_std_value = tl.load(input_ptr_inv_std + (block_index // 16), None, eviction_policy='evict_last')
    weight_value = tl.load(input_ptr_weight + (channel_index), None, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (channel_index), None, eviction_policy='evict_last')
    
    centered_input = mean_value - var_value
    scale_factor = 16384.0
    epsilon = 1e-05
    
    normalized_variance = inv_std_value / scale_factor
    adjusted_variance = normalized_variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    
    normalized_input = centered_input * reciprocal_sqrt
    scaled_input = normalized_input * weight_value
    output_value = scaled_input + bias_value
    
    tl.store(output_ptr + (index), output_value, None)