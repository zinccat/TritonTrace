# From: 28_HardSigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardsigmoid_0poi_fused_hardsigmoid_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_values = tl.load(in_ptr0 + (x0), xmask)
    bias = 3.0
    biased_input = input_values + bias
    lower_bound = 0.0
    clamped_input = triton_helpers.maximum(biased_input, lower_bound)
    upper_bound = 6.0
    hard_sigmoid_input = triton_helpers.minimum(clamped_input, upper_bound)
    scale_factor = 0.16666666666666666
    hard_sigmoid_output = hard_sigmoid_input * scale_factor
    tl.store(out_ptr0 + (x0), hard_sigmoid_output, xmask)