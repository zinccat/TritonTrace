# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_backward_threshold_backward_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    grad_output = tl.load(in_out_ptr0 + (x0), xmask)
    input_data = tl.load(in_ptr0 + (x0), xmask)
    lower_threshold = -3.0
    is_below_lower = grad_output < lower_threshold
    upper_threshold = 3.0
    is_below_upper = grad_output <= upper_threshold
    scale_factor = 0.3333333333333333
    scaled_grad_output = grad_output * scale_factor
    bias = 0.5
    adjusted_grad_output = scaled_grad_output + bias
    elementwise_product = input_data * adjusted_grad_output
    conditional_output = tl.where(is_below_upper, elementwise_product, input_data)
    zero = 0.0
    final_output = tl.where(is_below_lower, zero, conditional_output)
    tl.store(in_out_ptr0 + (x0), final_output, xmask)