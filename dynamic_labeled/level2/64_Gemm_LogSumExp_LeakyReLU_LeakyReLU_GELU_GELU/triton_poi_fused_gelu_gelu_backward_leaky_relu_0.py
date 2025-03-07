# From: 64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_leaky_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input0 = tl.load(in_ptr0 + (x0), xmask)
    input1 = tl.load(in_ptr1 + (x0), xmask)

    zero = 0.0
    leaky_relu_slope = 0.01
    half = 0.5
    sqrt_half = 0.7071067811865476
    one = 1.0
    erf_coeff = 0.3989422804014327

    # Leaky ReLU
    is_positive = input1 > zero
    leaky_relu_output = tl.where(is_positive, input1, input1 * leaky_relu_slope)

    # Second Leaky ReLU
    is_positive_leaky = leaky_relu_output > zero
    second_leaky_relu_output = tl.where(is_positive_leaky, leaky_relu_output, leaky_relu_output * leaky_relu_slope)

    # GELU
    gelu_input = second_leaky_relu_output * half
    erf_input = second_leaky_relu_output * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    gelu_intermediate = gelu_input * (erf_result + one)

    # Second GELU
    erf_input2 = gelu_intermediate * sqrt_half
    erf_result2 = tl.extra.cuda.libdevice.erf(erf_input2)
    gelu_output = (erf_result2 + one) * half

    # GELU approximation
    squared_gelu_intermediate = gelu_intermediate * gelu_intermediate
    exp_component = tl.math.exp(-0.5 * squared_gelu_intermediate)
    gelu_approx = gelu_intermediate + exp_component * erf_coeff * gelu_intermediate

    # Final GELU output
    final_gelu_output = gelu_output + gelu_approx

    # Backward pass
    output_gradient = input0 * final_gelu_output

    tl.store(out_ptr0 + (x0), output_gradient, xmask)