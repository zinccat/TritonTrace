# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_backward_threshold_backward_2poi_fused_hardswish_backward_threshold_backward_2(
    in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_grad = tl.load(in_out_ptr0 + (x0), xmask)
    input_data = tl.load(in_ptr0 + (x0), xmask)
    lower_threshold = -3.0
    is_below_lower = input_grad < lower_threshold
    upper_threshold = 3.0
    is_below_upper = input_grad <= upper_threshold
    scale_factor = 0.3333333333333333
    scaled_input = input_grad * scale_factor
    offset = 0.5
    activation = scaled_input + offset
    grad_through_activation = input_data * activation
    conditional_grad = tl.where(is_below_upper, grad_through_activation, input_data)
    zero_grad = 0.0
    final_grad = tl.where(is_below_lower, zero_grad, conditional_grad)
    tl.store(in_out_ptr0 + (x0), final_grad, xmask)