# From: 69_Conv2d_HardSwish_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_backward_threshold_backward_0poi_fused_hardswish_backward_threshold_backward_0(
    in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input and intermediate values
    grad_output = tl.load(in_out_ptr0 + (x0), xmask)
    relu_mask = tl.load(in_ptr0 + (x0), xmask).to(tl.int1)
    input_data = tl.load(in_ptr1 + (x0), xmask)

    # Define constants
    lower_bound = -3.0
    upper_bound = 3.0
    zero = 0.0
    scale_factor = 0.3333333333333333
    offset = 0.5

    # Compute conditions
    is_below_lower = grad_output < lower_bound
    is_within_bounds = grad_output <= upper_bound

    # Compute HardSwish backward
    relu_grad = tl.where(relu_mask, zero, input_data)
    scaled_grad = grad_output * scale_factor
    shifted_grad = scaled_grad + offset
    hardswish_grad = relu_grad * shifted_grad
    bounded_grad = tl.where(is_within_bounds, hardswish_grad, relu_grad)

    # Apply threshold
    final_grad = tl.where(is_below_lower, zero, bounded_grad)

    # Store the result
    tl.store(in_out_ptr0 + (x0), final_grad, xmask)