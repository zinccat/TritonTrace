# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_16poi_fused__native_batch_norm_legit_functional_16(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_offset, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 192
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    index_2d = indices
    index_1d = indices % 96
    
    mean_value = tl.load(input_ptr_mean + (index_2d), mask)
    variance_value = tl.load(input_ptr_var + (index_1d), mask, eviction_policy='evict_last')
    mean_value_1 = tl.load(input_ptr_mean + (index_1d), mask, eviction_policy='evict_last')
    mean_value_2 = tl.load(input_ptr_mean + (96 + index_1d), mask, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (index_1d), mask, eviction_policy='evict_last')
    offset_value = tl.load(input_ptr_offset + (index_1d), mask, eviction_policy='evict_last')
    
    normalized_value = mean_value - variance_value
    centered_value_1 = mean_value_1 - variance_value
    squared_centered_value_1 = centered_value_1 * centered_value_1
    centered_value_2 = mean_value_2 - variance_value
    squared_centered_value_2 = centered_value_2 * centered_value_2
    sum_of_squares = squared_centered_value_1 + squared_centered_value_2
    
    divisor = 2.0
    adjusted_variance = sum_of_squares / divisor
    epsilon = 1e-05
    variance_with_epsilon = adjusted_variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    
    scaled_normalized_value = normalized_value * reciprocal_sqrt
    scaled_value = scaled_normalized_value * scale_value
    output_value = scaled_value + offset_value
    
    tl.store(output_ptr + (index_2d), output_value, mask)