# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_sub_1poi_fused_hardswish_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input data
    input_data = tl.load(in_ptr0 + (x0), xmask)

    # HardSwish operation
    subtract_value = 0.5
    shifted_input = input_data - subtract_value
    add_value = 3.0
    shifted_input_plus_add = shifted_input + add_value
    zero_value = 0.0
    max_value = triton_helpers.maximum(shifted_input_plus_add, zero_value)
    min_value = 6.0
    clamped_value = triton_helpers.minimum(max_value, min_value)
    hardswish_result = shifted_input * clamped_value

    # Scale the result
    scale_factor = 0.16666666666666666
    scaled_result = hardswish_result * scale_factor

    # Store the result
    tl.store(out_ptr0 + (x0), scaled_result, xmask)