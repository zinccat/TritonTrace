# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_26poi_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_26(input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr1, output_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 864)
    
    input_value0 = tl.load(input_ptr0 + (x2), xmask)
    input_value1 = tl.load(input_ptr1 + (x0), xmask, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr0 + (x0), xmask, eviction_policy='evict_last')
    input_value6 = tl.load(input_ptr0 + (864 + x0), xmask, eviction_policy='evict_last')
    input_value16 = tl.load(input_ptr2 + (x0), xmask, eviction_policy='evict_last')
    input_value18 = tl.load(input_ptr3 + (x0), xmask, eviction_policy='evict_last')
    
    diff0 = input_value0 - input_value1
    diff3 = input_value3 - input_value1
    squared_diff3 = diff3 * diff3
    diff6 = input_value6 - input_value1
    squared_diff6 = diff6 * diff6
    sum_of_squares = squared_diff3 + squared_diff6
    
    divisor = 2.0
    epsilon = 1e-05
    normalized_sum = sum_of_squares / divisor
    adjusted_sum = normalized_sum + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_sum)
    
    scaled_diff = diff0 * reciprocal_sqrt
    weighted_scaled_diff = scaled_diff * input_value16
    final_value = weighted_scaled_diff + input_value18
    
    zero_tensor = tl.full([1], 0, tl.int32)
    max_value = triton_helpers.maximum(zero_tensor, final_value)
    normalized_max = max_value / 1.0
    zero_threshold = 0.0
    threshold_condition = max_value <= zero_threshold
    
    tl.store(output_ptr1 + (x2), normalized_max, xmask)
    tl.store(output_ptr2 + (x2), threshold_condition, xmask)