# From: 82_Conv2d_Tanh_Scaling_BiasAdd_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_tanh_tanh_backward_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    grad_output = tl.load(in_out_ptr0 + (x0), xmask)
    input_data = tl.load(in_ptr0 + (x0), xmask)
    scale_factor = 2.0
    scaled_grad_output = grad_output * scale_factor
    tanh_input = tl.extra.cuda.libdevice.tanh(input_data)
    tanh_squared = tanh_input * tanh_input
    one_minus_tanh_squared = 1.0 - tanh_squared
    grad_input = scaled_grad_output * one_minus_tanh_squared
    tl.store(in_out_ptr0 + (x0), grad_input, xmask)