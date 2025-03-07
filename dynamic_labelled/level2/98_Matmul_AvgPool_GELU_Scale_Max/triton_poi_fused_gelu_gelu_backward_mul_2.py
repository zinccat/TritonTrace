# From: 98_Matmul_AvgPool_GELU_Scale_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_mul_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_value = tl.load(in_ptr0 + (x0), xmask)
    grad_output = tl.load(in_out_ptr0 + (x0), xmask)

    scale_factor = 2.0
    scaled_input = input_value * scale_factor

    erf_scale = 0.7071067811865476
    scaled_grad_output = grad_output * erf_scale

    erf_result = tl.extra.cuda.libdevice.erf(scaled_grad_output)
    erf_offset = 1.0
    erf_adjusted = erf_result + erf_offset

    erf_half = 0.5
    erf_term = erf_adjusted * erf_half

    grad_output_squared = grad_output * grad_output
    exp_scale = -0.5
    exp_argument = grad_output_squared * exp_scale

    exp_result = tl.math.exp(exp_argument)
    exp_coefficient = 0.3989422804014327
    exp_term = exp_result * exp_coefficient

    gaussian_term = grad_output * exp_term
    gelu_derivative = erf_term + gaussian_term

    final_result = scaled_input * gelu_derivative

    tl.store(in_out_ptr0 + (x0), final_result, xmask)