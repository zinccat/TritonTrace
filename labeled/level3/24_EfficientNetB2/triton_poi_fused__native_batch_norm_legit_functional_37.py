# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_37poi_fused__native_batch_norm_legit_functional_37(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 576
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    global_indices = indices
    local_indices = indices % 288
    
    mean_value = tl.load(input_ptr_mean + (global_indices), mask)
    variance_value = tl.load(input_ptr_var + (local_indices), mask, eviction_policy='evict_last')
    mean_value_repeated = tl.load(input_ptr_mean + (local_indices), mask, eviction_policy='evict_last')
    variance_value_next = tl.load(input_ptr_mean + (288 + local_indices), mask, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (local_indices), mask, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (local_indices), mask, eviction_policy='evict_last')
    
    normalized_value = mean_value - variance_value
    centered_value = mean_value_repeated - variance_value
    squared_centered_value = centered_value * centered_value
    squared_next_value = variance_value_next - variance_value
    squared_next_value_squared = squared_next_value * squared_next_value
    sum_of_squares = squared_centered_value + squared_next_value_squared
    divisor = 2.0
    variance_adjusted = sum_of_squares / divisor
    epsilon = 1e-05
    variance_adjusted_epsilon = variance_adjusted + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)
    normalized_scaled_value = normalized_value * reciprocal_sqrt
    scaled_value = normalized_scaled_value * scale_value
    output_value = scaled_value + bias_value
    
    tl.store(output_ptr + (global_indices), output_value, mask)