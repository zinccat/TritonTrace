# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_47poi_fused_gelu_47(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_values = tl.load(in_ptr0 + (x0), xmask)
    half = 0.5
    scaled_input = input_values * half
    sqrt_half = 0.7071067811865476
    sqrt_half_scaled_input = input_values * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(sqrt_half_scaled_input)
    one = 1.0
    erf_plus_one = erf_result + one
    gelu_result = scaled_input * erf_plus_one
    tl.store(out_ptr0 + (x0), gelu_result, xmask)