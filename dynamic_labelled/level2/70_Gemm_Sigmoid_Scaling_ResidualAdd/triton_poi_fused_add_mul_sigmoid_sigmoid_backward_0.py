# From: 70_Gemm_Sigmoid_Scaling_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_mul_sigmoid_sigmoid_backward_0poi_fused_add_mul_sigmoid_sigmoid_backward_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    
    input_value = tl.load(in_ptr0 + (x0), xmask)
    output_value = tl.load(in_out_ptr0 + (x0), xmask)
    
    scale_factor = 2.0
    scaled_input = input_value * scale_factor
    
    sigmoid_output = tl.sigmoid(output_value)
    one_minus_sigmoid = 1.0 - sigmoid_output
    sigmoid_derivative = sigmoid_output * one_minus_sigmoid
    
    updated_value = scaled_input * sigmoid_derivative
    final_result = input_value + updated_value
    
    tl.store(in_out_ptr0 + (x0), final_result, xmask)