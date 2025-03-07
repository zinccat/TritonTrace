# From: 86_Matmul_Divide_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_gelu_gelu_backward_0poi_fused_div_gelu_gelu_backward_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_value = tl.load(in_ptr0 + (x0), xmask)
    grad_output = tl.load(in_out_ptr0 + (x0), xmask)

    scale_factor = 0.1
    scaled_grad_output = grad_output * scale_factor

    erf_coefficient = 0.7071067811865476
    scaled_erf_input = scaled_grad_output * erf_coefficient

    erf_result = tl.extra.cuda.libdevice.erf(scaled_erf_input)

    erf_offset = 1.0
    erf_adjusted = erf_result + erf_offset

    erf_half = 0.5
    erf_scaled = erf_adjusted * erf_half

    squared_scaled_grad_output = scaled_grad_output * scaled_grad_output

    exp_coefficient = -0.5
    exp_input = squared_scaled_grad_output * exp_coefficient

    exp_result = tl.math.exp(exp_input)

    gaussian_coefficient = 0.3989422804014327
    gaussian_term = exp_result * gaussian_coefficient

    gaussian_scaled = scaled_grad_output * gaussian_term

    gelu_result = erf_scaled + gaussian_scaled

    grad_input = input_value * gelu_result

    final_grad_output = grad_input * scale_factor

    tl.store(in_out_ptr0 + (x0), final_grad_output, xmask)