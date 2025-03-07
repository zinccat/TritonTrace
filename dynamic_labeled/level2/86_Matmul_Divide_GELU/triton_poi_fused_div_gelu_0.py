# From: 86_Matmul_Divide_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input data
    input_data = tl.load(in_ptr0 + (x0), xmask)

    # Constants
    scale_factor = 0.1
    half = 0.5
    sqrt_half = 0.7071067811865476
    one = 1.0

    # Computation
    scaled_input = input_data * scale_factor
    half_scaled_input = scaled_input * half
    sqrt_half_scaled_input = scaled_input * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(sqrt_half_scaled_input)
    erf_plus_one = erf_result + one
    final_result = half_scaled_input * erf_plus_one

    # Store result
    tl.store(out_ptr0 + (x0), final_result, xmask)