# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_19poi_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_19(input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr1, output_ptr2, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 1152
    offset = tl.program_id(0) * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    global_index = indices
    local_index = (indices % 576)
    
    input_value0 = tl.load(input_ptr0 + (global_index), mask)
    input_value1 = tl.load(input_ptr1 + (local_index), mask, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr0 + (local_index), mask, eviction_policy='evict_last')
    input_value6 = tl.load(input_ptr0 + (576 + local_index), mask, eviction_policy='evict_last')
    input_value16 = tl.load(input_ptr2 + (local_index), mask, eviction_policy='evict_last')
    input_value18 = tl.load(input_ptr3 + (local_index), mask, eviction_policy='evict_last')
    
    diff0 = input_value0 - input_value1
    diff3 = input_value3 - input_value1
    squared_diff3 = diff3 * diff3
    diff6 = input_value6 - input_value1
    squared_diff6 = diff6 * diff6
    sum_of_squares = squared_diff3 + squared_diff6
    
    divisor = 2.0
    normalized_sum = sum_of_squares / divisor
    epsilon = 1e-05
    adjusted_sum = normalized_sum + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_sum)
    
    scaled_diff = diff0 * reciprocal_sqrt
    scaled_output = scaled_diff * input_value16
    final_output = scaled_output + input_value18
    
    zero_tensor = tl.full([1], 0, tl.int32)
    max_output = triton_helpers.maximum(zero_tensor, final_output)
    normalized_max_output = max_output / 1.0
    zero_threshold = 0.0
    threshold_mask = max_output <= zero_threshold
    
    tl.store(output_ptr1 + (global_index), normalized_max_output, mask)
    tl.store(output_ptr2 + (global_index), threshold_mask, mask)