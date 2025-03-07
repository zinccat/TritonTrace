# From: 9_Matmul_Subtract_Multiply_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_mul_relu_sub_threshold_backward_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 5
    
    # Load input and intermediate results
    input_output_value = tl.load(in_out_ptr0 + (x2), xmask)
    input_value = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    
    # Perform addition
    added_value = input_output_value + input_value
    
    # Constants for subtraction and multiplication
    subtract_value = 2.0
    multiply_value = 1.5
    
    # Perform subtraction and multiplication
    subtracted_value = added_value - subtract_value
    multiplied_value = subtracted_value * multiply_value
    
    # ReLU operation
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, multiplied_value)
    
    # Threshold comparison
    threshold_value = 0.0
    threshold_result = relu_output <= threshold_value
    
    # Store results
    tl.store(in_out_ptr0 + (x2), relu_output, xmask)
    tl.store(out_ptr0 + (x2), threshold_result, xmask)