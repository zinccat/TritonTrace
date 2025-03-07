# From: 69_Conv2d_HardSwish_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_backward_threshold_backward_0(
    in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load data from input pointers
    grad_output = tl.load(in_out_ptr0 + (x0), xmask)
    input_tensor = tl.load(in_ptr0 + (x0), xmask).to(tl.int1)
    threshold_tensor = tl.load(in_ptr1 + (x0), xmask)

    # Define constants
    lower_bound = -3.0
    upper_bound = 3.0
    zero = 0.0
    scale_factor = 0.3333333333333333
    offset = 0.5

    # Compute conditions
    is_below_lower_bound = grad_output < lower_bound
    is_within_bounds = grad_output <= upper_bound

    # Compute intermediate values
    thresholded_value = tl.where(input_tensor, zero, threshold_tensor)
    scaled_output = grad_output * scale_factor
    adjusted_output = scaled_output + offset
    hardswish_output = thresholded_value * adjusted_output

    # Apply conditions to compute final gradient
    final_gradient = tl.where(is_within_bounds, hardswish_output, thresholded_value)
    final_gradient = tl.where(is_below_lower_bound, zero, final_gradient)

    # Store the result back to the output pointer
    tl.store(in_out_ptr0 + (x0), final_gradient, xmask)