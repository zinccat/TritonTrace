# From: 53_Gemm_Scaling_Hardtanh_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_gelu_hardtanh_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    input_indices = xindex
    input_values = tl.load(in_ptr0 + (input_indices), None)
    half = 0.5
    scaled_input = input_values * half
    lower_bound = -2.0
    clamped_value = triton_helpers.maximum(scaled_input, lower_bound)
    upper_bound = 2.0
    hardtanh_value = triton_helpers.minimum(clamped_value, upper_bound)
    scaled_hardtanh = hardtanh_value * half
    erf_scale = 0.7071067811865476
    erf_input = hardtanh_value * erf_scale
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    one = 1.0
    gelu_input = erf_result + one
    gelu_output = scaled_hardtanh * gelu_input
    tl.store(out_ptr0 + (input_indices), gelu_output, None)