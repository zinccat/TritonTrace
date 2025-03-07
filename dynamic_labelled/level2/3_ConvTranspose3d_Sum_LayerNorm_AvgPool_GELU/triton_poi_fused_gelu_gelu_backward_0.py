# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_0poi_fused_gelu_gelu_backward_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    
    input_value = tl.load(in_ptr0 + (x0), None)
    grad_output = tl.load(in_out_ptr0 + (x0), None)
    
    sqrt_2 = 0.7071067811865476
    scaled_grad_output = grad_output * sqrt_2
    
    erf_result = tl.extra.cuda.libdevice.erf(scaled_grad_output)
    one = 1.0
    erf_plus_one = erf_result + one
    
    half = 0.5
    half_erf_plus_one = erf_plus_one * half
    
    grad_output_squared = grad_output * grad_output
    neg_half = -0.5
    exp_term = grad_output_squared * neg_half
    
    exp_result = tl.math.exp(exp_term)
    inv_sqrt_2pi = 0.3989422804014327
    gaussian_term = exp_result * inv_sqrt_2pi
    
    grad_output_gaussian = grad_output * gaussian_term
    gelu_derivative = half_erf_plus_one + grad_output_gaussian
    
    grad_input = input_value * gelu_derivative
    
    tl.store(in_out_ptr0 + (x0), grad_input, None)