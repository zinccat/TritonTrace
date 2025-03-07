# From: 90_Conv3d_LeakyReLU_Sum_Clamp_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_value = tl.load(in_out_ptr0 + (x0), xmask)
    input_grad = tl.load(in_ptr0 + (x0), xmask)
    zero_threshold = 0.0
    is_positive = input_value > zero_threshold
    negative_slope = 0.2
    negative_value = input_grad * negative_slope
    output_grad = tl.where(is_positive, input_grad, negative_value)
    tl.store(in_out_ptr0 + (x0), output_grad, xmask)