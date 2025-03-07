# From: 31_ELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_elu_0poi_fused_elu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_indices = xindex
    input_values = tl.load(in_ptr0 + (x_indices), xmask)
    zero_threshold = 0.0
    is_positive = input_values > zero_threshold
    one_multiplier = 1.0
    positive_values = input_values * one_multiplier
    expm1_values = tl.extra.cuda.libdevice.expm1(positive_values)
    scaled_expm1 = expm1_values * one_multiplier
    elu_output = tl.where(is_positive, positive_values, scaled_expm1)
    tl.store(out_ptr0 + (x_indices), elu_output, xmask)