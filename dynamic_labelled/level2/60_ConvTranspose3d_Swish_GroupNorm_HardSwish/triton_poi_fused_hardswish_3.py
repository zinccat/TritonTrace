# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_3poi_fused_hardswish_3(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < xnumel
    x0 = x_index
    input_value = tl.load(in_out_ptr0 + (x0), x_mask)
    bias_value = 3.0
    biased_input = input_value + bias_value
    lower_bound = 0.0
    clamped_input = triton_helpers.maximum(biased_input, lower_bound)
    upper_bound = 6.0
    clipped_input = triton_helpers.minimum(clamped_input, upper_bound)
    scaled_input = input_value * clipped_input
    scaling_factor = 0.16666666666666666
    output_value = scaled_input * scaling_factor
    tl.store(in_out_ptr0 + (x0), output_value, x_mask)