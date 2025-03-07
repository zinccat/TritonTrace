# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mish_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    input_indices = xindex
    input_values = tl.load(in_ptr0 + (input_indices), xmask)
    threshold = 20.0
    is_greater_than_threshold = input_values > threshold
    exp_values = tl.math.exp(input_values)
    log1p_values = tl.extra.cuda.libdevice.log1p(exp_values)
    adjusted_values = tl.where(is_greater_than_threshold, input_values, log1p_values)
    tanh_values = tl.extra.cuda.libdevice.tanh(adjusted_values)
    mish_output = input_values * tanh_values
    tl.store(out_ptr0 + (input_indices), mish_output, xmask)