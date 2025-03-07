# From: 33_BatchNorm

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_2(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    index = block_indices
    channel_index = (block_indices // 65536) % 64
    input_value = tl.load(input_ptr_mean + (index), None)
    mean_value = tl.load(input_ptr_mean + (channel_index), None, eviction_policy='evict_last')
    var_value = tl.load(input_ptr_var + (channel_index), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (channel_index), None, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (channel_index), None, eviction_policy='evict_last')
    
    centered_value = input_value - mean_value
    variance_scale = 1048576.0
    normalized_variance = var_value / variance_scale
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    normalized_input = centered_value * inv_sqrt_variance
    scaled_input = normalized_input * scale_value
    output_value = scaled_input + bias_value
    
    tl.store(output_ptr + (index), output_value, None)