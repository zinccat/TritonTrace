# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_18poi_fused_gelu_18(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = x_index
    input_value = tl.load(in_ptr0 + (x0), None)
    half = 0.5
    scaled_input = input_value * half
    erf_coefficient = 0.7071067811865476
    erf_input = input_value * erf_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    one = 1.0
    erf_adjusted = erf_result + one
    gelu_output = scaled_input * erf_adjusted
    tl.store(out_ptr0 + (x0), gelu_output, None)