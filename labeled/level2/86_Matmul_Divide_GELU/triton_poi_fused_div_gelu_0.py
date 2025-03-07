# From: 86_Matmul_Divide_GELU

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_div_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    input_indices = xindex
    input_values = tl.load(in_ptr0 + (input_indices), None)
    scale_factor = 0.1
    scaled_values = input_values * scale_factor
    half = 0.5
    scaled_half = scaled_values * half
    sqrt_half = 0.7071067811865476
    sqrt_scaled_values = scaled_values * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(sqrt_scaled_values)
    one = 1.0
    erf_plus_one = erf_result + one
    final_result = scaled_half * erf_plus_one
    tl.store(out_ptr0 + (input_indices), final_result, None)