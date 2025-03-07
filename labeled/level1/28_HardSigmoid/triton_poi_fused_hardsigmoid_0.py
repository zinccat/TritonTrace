# From: 28_HardSigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_hardsigmoid_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    input_index = x_index
    input_value = tl.load(in_ptr0 + (input_index), None)
    bias = 3.0
    biased_input = input_value + bias
    lower_bound = 0.0
    clamped_value = triton_helpers.maximum(biased_input, lower_bound)
    upper_bound = 6.0
    hard_sigmoid_value = triton_helpers.minimum(clamped_value, upper_bound)
    scale_factor = 0.16666666666666666
    scaled_output = hard_sigmoid_value * scale_factor
    tl.store(out_ptr0 + (input_index), scaled_output, None)