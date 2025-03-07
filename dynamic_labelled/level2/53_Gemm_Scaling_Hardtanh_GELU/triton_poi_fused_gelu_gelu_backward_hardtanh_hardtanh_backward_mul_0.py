# From: 53_Gemm_Scaling_Hardtanh_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_hardtanh_hardtanh_backward_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input data
    input_data = tl.load(in_out_ptr0 + (x0), xmask)
    input_ptr_data = tl.load(in_ptr0 + (x0), xmask)

    # Constants
    half = 0.5
    neg_two = -2.0
    two = 2.0
    sqrt_half = 0.7071067811865476
    one = 1.0
    inv_sqrt_two = 0.3989422804014327

    # Compute intermediate values
    scaled_input = input_data * half
    lower_bound = scaled_input <= neg_two
    upper_bound = scaled_input >= two
    out_of_bounds = lower_bound | upper_bound

    clamped_input = triton_helpers.maximum(scaled_input, neg_two)
    clamped_input = triton_helpers.minimum(clamped_input, two)
    scaled_clamped_input = clamped_input * sqrt_half

    erf_result = tl.extra.cuda.libdevice.erf(scaled_clamped_input)
    erf_scaled = (erf_result + one) * half

    squared_clamped_input = clamped_input * clamped_input
    exp_component = -0.5 * squared_clamped_input
    exp_result = tl.math.exp(exp_component)
    gaussian_component = exp_result * inv_sqrt_two
    gaussian_scaled = clamped_input * gaussian_component

    gelu_result = erf_scaled + gaussian_scaled
    scaled_gelu = input_ptr_data * gelu_result

    # Apply mask and store result
    masked_result = tl.where(out_of_bounds, 0.0, scaled_gelu)
    final_result = masked_result * half
    tl.store(in_out_ptr0 + (x0), final_result, xmask)