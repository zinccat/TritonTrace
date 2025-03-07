# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_23poi_fused__native_batch_norm_legit_functional_23(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 288
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    global_indices = indices
    channel_indices = indices % 144
    
    mean = tl.load(input_ptr_mean + (global_indices), mask)
    variance = tl.load(input_ptr_var + (channel_indices), mask, eviction_policy='evict_last')
    mean_channel = tl.load(input_ptr_mean + (channel_indices), mask, eviction_policy='evict_last')
    variance_channel = tl.load(input_ptr_mean + (144 + channel_indices), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_indices), mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_indices), mask, eviction_policy='evict_last')
    
    normalized = mean - variance
    centered_mean = mean_channel - variance
    centered_mean_squared = centered_mean * centered_mean
    centered_variance = variance_channel - variance
    centered_variance_squared = centered_variance * centered_variance
    sum_of_squares = centered_mean_squared + centered_variance_squared
    divisor = 2.0
    variance_sum = sum_of_squares / divisor
    epsilon = 1e-05
    variance_sum_epsilon = variance_sum + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_sum_epsilon)
    normalized_scaled = normalized * inv_sqrt_variance
    scaled = normalized_scaled * scale
    output = scaled + bias
    
    tl.store(output_ptr + (global_indices), output, mask)