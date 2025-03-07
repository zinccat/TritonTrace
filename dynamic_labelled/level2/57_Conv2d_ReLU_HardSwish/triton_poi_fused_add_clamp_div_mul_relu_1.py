# From: 57_Conv2d_ReLU_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_div_mul_relu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_values = tl.load(in_ptr0 + (x0), xmask)
    zero_value = tl.full([1], 0, tl.int32)
    max_with_zero = triton_helpers.maximum(zero_value, input_values)

    add_constant = 3.0
    added_values = max_with_zero + add_constant

    multiply_constant = 0.16666666666666666
    multiplied_values = added_values * multiply_constant

    clamp_min = 0.0
    max_with_clamp_min = triton_helpers.maximum(multiplied_values, clamp_min)

    clamp_max = 1.0
    clamped_values = triton_helpers.minimum(max_with_clamp_min, clamp_max)

    final_output = max_with_zero * clamped_values
    tl.store(out_ptr0 + (x0), final_output, xmask)