# From: 57_Conv2d_ReLU_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_div_mul_relu_1poi_fused_add_clamp_div_mul_relu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input values
    input_values = tl.load(in_ptr0 + (x0), xmask)

    # Initialize constants
    zero = tl.full([1], 0, tl.int32)
    three = 3.0
    one_sixth = 0.16666666666666666
    zero_float = 0.0
    one = 1.0

    # Apply ReLU (max with zero)
    relu_output = triton_helpers.maximum(zero, input_values)

    # Add constant and scale
    add_result = relu_output + three
    scaled_result = add_result * one_sixth

    # Clamp between 0 and 1
    clamped_result = triton_helpers.maximum(scaled_result, zero_float)
    clamped_result = triton_helpers.minimum(clamped_result, one)

    # Multiply with original ReLU output
    final_result = relu_output * clamped_result

    # Store the result
    tl.store(out_ptr0 + (x0), final_result, xmask)