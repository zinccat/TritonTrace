# From: 53_Gemm_Scaling_Hardtanh_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_hardtanh_mul_0poi_fused_gelu_hardtanh_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input data
    input_data = tl.load(in_ptr0 + (x0), xmask)

    # Calculate 0.5 * input_data
    half_input = input_data * 0.5

    # Apply Hardtanh: max(-2.0, min(half_input, 2.0))
    hardtanh_min = -2.0
    hardtanh_max = 2.0
    hardtanh_applied = triton_helpers.minimum(triton_helpers.maximum(half_input, hardtanh_min), hardtanh_max)

    # Calculate 0.5 * hardtanh_applied
    half_hardtanh = hardtanh_applied * 0.5

    # Calculate sqrt(0.5) * hardtanh_applied
    sqrt_half = 0.7071067811865476
    sqrt_half_hardtanh = hardtanh_applied * sqrt_half

    # Calculate erf(sqrt_half_hardtanh)
    erf_result = tl.extra.cuda.libdevice.erf(sqrt_half_hardtanh)

    # Calculate 0.5 * (1.0 + erf_result)
    gelu_result = half_hardtanh * (erf_result + 1.0)

    # Store the result
    tl.store(out_ptr0 + (x0), gelu_result, xmask)