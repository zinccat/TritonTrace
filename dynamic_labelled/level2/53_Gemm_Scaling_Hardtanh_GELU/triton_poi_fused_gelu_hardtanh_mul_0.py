# From: 53_Gemm_Scaling_Hardtanh_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_hardtanh_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input values
    input_values = tl.load(in_ptr0 + (x0), xmask)

    # Calculate 0.5 * input_values
    half_input_values = input_values * 0.5

    # Apply Hardtanh with range [-2, 2]
    lower_bound = -2.0
    upper_bound = 2.0
    hardtanh_values = triton_helpers.maximum(half_input_values, lower_bound)
    hardtanh_values = triton_helpers.minimum(hardtanh_values, upper_bound)

    # Calculate 0.5 * hardtanh_values
    half_hardtanh_values = hardtanh_values * 0.5

    # Calculate sqrt(0.5) * hardtanh_values
    sqrt_half = 0.7071067811865476
    sqrt_half_hardtanh_values = hardtanh_values * sqrt_half

    # Calculate erf(sqrt_half_hardtanh_values)
    erf_values = tl.extra.cuda.libdevice.erf(sqrt_half_hardtanh_values)

    # Calculate 0.5 * (1 + erf_values)
    gelu_values = half_hardtanh_values * (erf_values + 1.0)

    # Store the result
    tl.store(out_ptr0 + (x0), gelu_values, xmask)