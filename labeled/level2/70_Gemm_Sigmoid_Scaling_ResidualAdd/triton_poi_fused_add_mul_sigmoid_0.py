# From: 70_Gemm_Sigmoid_Scaling_ResidualAdd

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_mul_sigmoid_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x_indices = x_indices
    input_values = tl.load(in_ptr0 + (x_indices), None)
    sigmoid_values = tl.sigmoid(input_values)
    scaling_factor = 2.0
    scaled_sigmoid = sigmoid_values * scaling_factor
    result_values = scaled_sigmoid + input_values
    tl.store(out_ptr0 + (x_indices), result_values, None)