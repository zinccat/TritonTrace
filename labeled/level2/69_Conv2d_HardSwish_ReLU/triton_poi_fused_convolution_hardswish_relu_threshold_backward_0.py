# From: 69_Conv2d_HardSwish_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_convolution_hardswish_relu_threshold_backward_0(
    in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Descriptive variable names
    batch_index = xindex
    channel_index = (xindex // 900) % 16
    batch_offset = (xindex // 14400)
    spatial_index = xindex % 14400
    
    # Load input and intermediate values
    input_value = tl.load(in_out_ptr0 + (batch_index), None)
    weight_value = tl.load(in_ptr0 + (channel_index), None, eviction_policy='evict_last')
    
    # Compute intermediate values
    sum_value = input_value + weight_value
    bias_value = 3.0
    biased_sum = sum_value + bias_value
    relu_min = 0.0
    relu_max = 6.0
    
    # Apply ReLU and HardSwish
    relu_applied = triton_helpers.maximum(biased_sum, relu_min)
    hardswish_applied = triton_helpers.minimum(relu_applied, relu_max)
    hardswish_result = sum_value * hardswish_applied
    
    # Final scaling
    scale_factor = 0.16666666666666666
    scaled_result = hardswish_result * scale_factor
    zero_tensor = tl.full([1], 0, tl.int32)
    max_result = triton_helpers.maximum(zero_tensor, scaled_result)
    
    # Threshold comparison
    threshold = relu_min
    is_below_threshold = max_result <= threshold
    
    # Store results
    tl.store(in_out_ptr0 + (batch_index), sum_value, None)
    tl.store(out_ptr0 + (batch_index), max_result, None)
    tl.store(out_ptr1 + (spatial_index + (14464 * batch_offset)), is_below_threshold, None)