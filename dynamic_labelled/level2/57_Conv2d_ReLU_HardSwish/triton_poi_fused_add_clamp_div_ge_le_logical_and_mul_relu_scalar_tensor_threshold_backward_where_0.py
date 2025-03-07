# From: 57_Conv2d_ReLU_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_div_ge_le_logical_and_mul_relu_scalar_tensor_threshold_backward_where_0(
    in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input and output tensors
    input_tensor = tl.load(in_out_ptr0 + (x0), xmask)
    weight_tensor = tl.load(in_ptr0 + (x0), xmask)

    # Initialize constants
    zero_int = tl.full([1], 0, tl.int32)
    zero_float = 0.0
    three_float = 3.0
    one_float = 1.0
    one_sixth_float = 0.16666666666666666

    # Apply ReLU and HardSwish operations
    relu_output = triton_helpers.maximum(zero_int, input_tensor)
    is_non_positive = relu_output <= zero_float

    hard_swish_input = relu_output + three_float
    scaled_hard_swish_input = hard_swish_input * one_sixth_float
    clamped_hard_swish_input = triton_helpers.maximum(scaled_hard_swish_input, zero_float)
    min_hard_swish_input = triton_helpers.minimum(clamped_hard_swish_input, one_float)

    weighted_input = weight_tensor * min_hard_swish_input
    is_in_hard_swish_range = (scaled_hard_swish_input >= zero_float) & (scaled_hard_swish_input <= one_float)
    hard_swish_output = tl.where(is_in_hard_swish_range, weight_tensor * relu_output, zero_float)
    scaled_hard_swish_output = hard_swish_output * one_sixth_float

    # Combine results
    combined_output = weighted_input + scaled_hard_swish_output
    final_output = tl.where(is_non_positive, zero_float, combined_output)

    # Store the result back to the output tensor
    tl.store(in_out_ptr0 + (x0), final_output, xmask)