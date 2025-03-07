# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input data
    input_data = tl.load(in_ptr0 + (x0), xmask)

    # HardSwish operation
    subtract_value = 0.5
    shifted_input = input_data - subtract_value
    upper_bound = 3.0
    bounded_input = shifted_input + upper_bound
    lower_bound = 0.0
    max_input = triton_helpers.maximum(bounded_input, lower_bound)
    upper_limit = 6.0
    min_input = triton_helpers.minimum(max_input, upper_limit)

    # Multiply and scale
    scaled_output = shifted_input * min_input
    scale_factor = 0.16666666666666666
    final_output = scaled_output * scale_factor

    # Store the result
    tl.store(out_ptr0 + (x0), final_output, xmask)