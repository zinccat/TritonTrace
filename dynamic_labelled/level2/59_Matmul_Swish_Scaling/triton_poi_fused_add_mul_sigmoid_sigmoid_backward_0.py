# From: 59_Matmul_Swish_Scaling

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
    scaled_sigmoid_output = scaled_input * sigmoid_output

    scaled_output_value = scaled_input * output_value

    one = 1.0
    one_minus_sigmoid_output = one - sigmoid_output
    sigmoid_derivative = sigmoid_output * one_minus_sigmoid_output

    gradient = scaled_output_value * sigmoid_derivative
    fused_result = scaled_sigmoid_output + gradient

    tl.store(in_out_ptr0 + (x0), fused_result, xmask)