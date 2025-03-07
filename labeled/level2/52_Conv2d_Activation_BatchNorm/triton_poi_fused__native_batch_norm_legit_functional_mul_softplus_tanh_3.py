# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_mul_softplus_tanh_3(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_bias, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    channel_index = (block_indices // 900) % 16
    
    mean_value = tl.load(input_ptr_mean + (index), None)
    variance_value = tl.load(input_ptr_var + (channel_index), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (channel_index), None, eviction_policy='evict_last')
    shift_value = tl.load(input_ptr_shift + (channel_index), None, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (channel_index), None, eviction_policy='evict_last')
    
    threshold = 20.0
    is_greater_than_threshold = input_data > threshold
    exp_value = tl.math.exp(input_data)
    log1p_value = tl.extra.cuda.libdevice.log1p(exp_value)
    softplus_value = tl.where(is_greater_than_threshold, input_data, log1p_value)
    tanh_value = tl.extra.cuda.libdevice.tanh(softplus_value)
    mul_value = tanh_value * input_data
    
    normalized_value = mul_value - mean_value
    variance_scale = 115200.0
    normalized_variance = variance_value / variance_scale
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_value = normalized_value * inv_sqrt_variance
    scaled_and_shifted_value = scaled_value * scale_value
    final_output = scaled_and_shifted_value + shift_value + bias_value
    
    tl.store(output_ptr + (index), final_output, None)