# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_63poi_fused__native_batch_norm_legit_functional_63(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    input_indices = block_indices
    channel_indices = input_indices % 320
    
    mean_value = tl.load(input_ptr_mean + (input_indices), None)
    variance_value = tl.load(input_ptr_var + (channel_indices), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (channel_indices), None, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (channel_indices), None, eviction_policy='evict_last')
    input_value = tl.load(input_ptr_input + (input_indices), None)
    
    normalized_input = input_value - mean_value
    variance_normalized = 640.0
    epsilon = 1e-05
    variance_adjusted = variance_value / variance_normalized
    variance_adjusted_epsilon = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)
    
    scaled_normalized_input = normalized_input * inv_sqrt_variance
    scaled_input = scaled_normalized_input * scale_value
    output_value = scaled_input + bias_value
    
    tl.store(output_ptr + (input_indices), output_value, None)