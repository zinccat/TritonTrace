# From: 88_MinGPTNewGelu

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_mul_pow_tanh_0poi_fused_add_mul_pow_tanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_value = tl.load(in_ptr0 + (x0), xmask)
    half = 0.5
    half_input = input_value * half
    input_squared = input_value * input_value
    input_cubed = input_squared * input_value
    gelu_coefficient = 0.044715
    gelu_term = input_cubed * gelu_coefficient
    gelu_input = input_value + gelu_term
    sqrt_2_over_pi = 0.7978845608028654
    scaled_gelu_input = gelu_input * sqrt_2_over_pi
    tanh_result = tl.extra.cuda.libdevice.tanh(scaled_gelu_input)
    one = 1.0
    tanh_plus_one = tanh_result + one
    final_result = half_input * tanh_plus_one

    tl.store(out_ptr0 + (x0), final_result, xmask)