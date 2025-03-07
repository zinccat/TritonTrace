# From: 26_GELU_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_0poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_values = tl.load(in_ptr0 + (x0), xmask)
    half = 0.5
    scaled_input = input_values * half
    erf_coefficient = 0.7071067811865476
    erf_input = input_values * erf_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    one = 1.0
    erf_adjusted = erf_result + one
    gelu_output = scaled_input * erf_adjusted
    tl.store(out_ptr0 + (x0), gelu_output, xmask)