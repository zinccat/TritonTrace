# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_backward_relu_threshold_backward_6(
    in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_value = tl.load(in_out_ptr0 + (x0), xmask)
    input_ptr_value = tl.load(in_ptr0 + (x0), xmask)
    zero_value = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_value, input_value)
    threshold_value = 0.0
    below_threshold = relu_output <= threshold_value
    thresholded_output = tl.where(below_threshold, threshold_value, input_ptr_value)
    tl.store(in_out_ptr0 + (x0), thresholded_output, xmask)