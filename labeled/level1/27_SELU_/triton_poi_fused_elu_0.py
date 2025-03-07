# From: 27_SELU_

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_elu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    input_indices = xindex
    input_values = tl.load(in_ptr0 + (input_indices), None)
    zero_threshold = 0.0
    greater_than_zero = input_values > zero_threshold
    alpha = 1.0507009873554805
    scaled_input = input_values * alpha
    one = 1.0
    input_times_one = input_values * one
    expm1_result = tl.extra.cuda.libdevice.expm1(input_times_one)
    beta = 1.7580993408473766
    elu_result = expm1_result * beta
    elu_output = tl.where(greater_than_zero, scaled_input, elu_result)
    tl.store(out_ptr0 + (input_indices), elu_output, None)