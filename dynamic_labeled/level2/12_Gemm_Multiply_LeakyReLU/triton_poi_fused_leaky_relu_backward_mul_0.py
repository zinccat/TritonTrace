# From: 12_Gemm_Multiply_LeakyReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_leaky_relu_backward_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    
    input_grad = tl.load(in_ptr0 + (x0), xmask).to(tl.int1)
    input_data = tl.load(in_ptr1 + (x0), xmask)
    
    negative_slope = 0.1
    negative_part = input_data * negative_slope
    
    leaky_relu_grad = tl.where(input_grad, input_data, negative_part)
    
    scale_factor = 2.0
    scaled_grad = leaky_relu_grad * scale_factor
    
    tl.store(out_ptr0 + (x0), scaled_grad, xmask)