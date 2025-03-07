# From: 1_Conv2D_ReLU_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_convolution_relu_threshold_backward_0(
    input_grad_ptr, weight_ptr, bias_ptr, output_grad_ptr, relu_mask_ptr, 
    xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    batch_index = (xindex // 900) % 16
    channel_index = xindex // 14400
    spatial_index = xindex % 14400
    
    # Load input gradient and weight
    input_grad = tl.load(input_grad_ptr + (xindex), None)
    weight = tl.load(weight_ptr + (batch_index), None, eviction_policy='evict_last')
    bias = tl.load(bias_ptr + (batch_index), None, eviction_policy='evict_last')
    
    # Compute convolution and add bias
    conv_result = input_grad + weight
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_result = triton_helpers.maximum(zero_tensor, conv_result)
    output = relu_result + bias
    
    # Compute ReLU mask
    relu_mask = relu_result <= 0.0
    
    # Store results
    tl.store(output_grad_ptr + (xindex), output, None)
    tl.store(relu_mask_ptr + (spatial_index + (14464 * channel_index)), relu_mask, None)