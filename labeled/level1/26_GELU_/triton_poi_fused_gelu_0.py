# From: 26_GELU_

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    input_indices = xindex
    input_values = tl.load(in_ptr0 + (input_indices), None)
    half = 0.5
    scaled_input = input_values * half
    sqrt_two_over_sqrt_pi = 0.7071067811865476
    erf_argument = input_values * sqrt_two_over_sqrt_pi
    erf_result = tl.extra.cuda.libdevice.erf(erf_argument)
    one = 1.0
    erf_plus_one = erf_result + one
    gelu_output = scaled_input * erf_plus_one
    tl.store(out_ptr0 + (input_indices), gelu_output, None)