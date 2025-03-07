# From: 26_GELU_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_indices = xindex

    input_values = tl.load(in_ptr0 + (x_indices), xmask)
    half = 0.5
    scaled_input = input_values * half

    sqrt_two_over_pi = 0.7071067811865476
    scaled_input_sqrt_two_over_pi = input_values * sqrt_two_over_pi

    erf_result = tl.extra.cuda.libdevice.erf(scaled_input_sqrt_two_over_pi)
    one = 1.0
    erf_plus_one = erf_result + one

    gelu_result = scaled_input * erf_plus_one
    tl.store(out_ptr0 + (x_indices), gelu_result, xmask)