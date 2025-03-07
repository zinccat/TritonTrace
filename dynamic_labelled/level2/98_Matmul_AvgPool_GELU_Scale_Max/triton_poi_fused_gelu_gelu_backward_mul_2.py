# From: 98_Matmul_AvgPool_GELU_Scale_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_mul_2poi_fused_gelu_gelu_backward_mul_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_value = tl.load(in_ptr0 + (x0), xmask)
    output_value = tl.load(in_out_ptr0 + (x0), xmask)

    scale_factor = 2.0
    scaled_input = input_value * scale_factor

    erf_scale = 0.7071067811865476
    scaled_output = output_value * erf_scale

    erf_result = tl.extra.cuda.libdevice.erf(scaled_output)
    erf_offset = 1.0
    erf_adjusted = erf_result + erf_offset

    erf_half = 0.5
    erf_half_adjusted = erf_adjusted * erf_half

    output_squared = output_value * output_value
    exp_scale = -0.5
    exp_argument = output_squared * exp_scale

    exp_result = tl.math.exp(exp_argument)
    exp_coefficient = 0.3989422804014327
    exp_scaled = exp_result * exp_coefficient

    output_exp_scaled = output_value * exp_scaled

    gelu_result = erf_half_adjusted + output_exp_scaled

    final_result = scaled_input * gelu_result

    tl.store(in_out_ptr0 + (x0), final_result, xmask)