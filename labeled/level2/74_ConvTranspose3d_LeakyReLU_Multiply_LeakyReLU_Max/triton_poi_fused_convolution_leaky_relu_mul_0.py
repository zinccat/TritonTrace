# From: 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_leaky_relu_mul_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    x3 = xindex
    x1 = (xindex // 131072) % 32
    
    # Load input data
    input_out = tl.load(in_out_ptr0 + (x3), None)
    input_0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    input_1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    
    # Perform addition
    added_result = input_out + input_0
    
    # Leaky ReLU parameters
    zero = 0.0
    leaky_relu_slope = 0.2
    
    # Apply Leaky ReLU
    positive_mask = added_result > zero
    leaky_relu_result = tl.where(positive_mask, added_result, added_result * leaky_relu_slope)
    
    # Multiply with input_1
    multiplied_result = leaky_relu_result * input_1
    
    # Apply Leaky ReLU again
    positive_mask_2 = multiplied_result > zero
    final_result = tl.where(positive_max, multiplied_result, multiplied_result * leaky_relu_slope)
    
    # Store results
    tl.store(in_out_ptr0 + (x3), added_result, None)
    tl.store(out_ptr0 + (x3), final_result, None)