# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_3(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input,
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    linear_index = block_indices
    channel_index = (block_indices // 250047) % 16
    spatial_index = block_indices % 3969
    batch_index = (block_indices // 3969)
    
    mean_value = tl.load(input_ptr_mean + (linear_index), None)
    variance_value = tl.load(input_ptr_var + (channel_index), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (channel_index), None, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (channel_index), None, eviction_policy='evict_last')
    input_value = tl.load(input_ptr_input + (linear_index), None)
    
    normalized_input = input_value - mean_value
    variance_epsilon = 32006016.0
    variance_adjusted = variance_value / variance_epsilon
    epsilon = 1e-05
    variance_adjusted_with_epsilon = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_with_epsilon)
    normalized_scaled_input = normalized_input * inv_sqrt_variance
    scaled_input = normalized_scaled_input * scale_value
    output_value = scaled_input + bias_value
    
    tl.store(output_ptr + (spatial_index + (4000 * batch_index)), output_value, None)