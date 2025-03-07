# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_backward_sub_2poi_fused_hardswish_backward_sub_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_grad = tl.load(in_out_ptr0 + (x0), xmask)
    input_data = tl.load(in_ptr0 + (x0), xmask)
    half = 0.5
    subtracted_value = input_grad - half
    lower_bound = -3.0
    upper_bound = 3.0
    one_third = 0.3333333333333333
    scaled_value = subtracted_value * one_third
    adjusted_value = scaled_value + half
    hardswish_grad = input_data * adjusted_value
    conditional_grad = tl.where(subtracted_value <= upper_bound, hardswish_grad, input_data)
    zero_grad = 0.0
    final_grad = tl.where(subtracted_value < lower_bound, zero_grad, conditional_grad)
    tl.store(in_out_ptr0 + (x0), final_grad, xmask)