# From: 28_HardSigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardsigmoid_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < xnumel
    x_indices = x_index
    input_values = tl.load(in_ptr0 + (x_indices), x_mask)
    bias = 3.0
    biased_input = input_values + bias
    lower_bound = 0.0
    max_value = triton_helpers.maximum(biased_input, lower_bound)
    upper_bound = 6.0
    clamped_value = triton_helpers.minimum(max_value, upper_bound)
    scale_factor = 0.16666666666666666
    output_values = clamped_value * scale_factor
    tl.store(out_ptr0 + (x_indices), output_values, x_mask)