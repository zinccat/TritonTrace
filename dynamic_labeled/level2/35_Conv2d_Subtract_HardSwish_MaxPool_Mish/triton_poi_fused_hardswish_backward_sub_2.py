# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_backward_sub_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input and output tensors
    input_tensor = tl.load(in_out_ptr0 + (x0), xmask)
    weight_tensor = tl.load(in_ptr0 + (x0), xmask)

    # Constants
    half = 0.5
    lower_bound = -3.0
    upper_bound = 3.0
    one_third = 0.3333333333333333

    # Compute intermediate values
    diff = input_tensor - half
    is_below_lower = diff < lower_bound
    is_within_bounds = diff <= upper_bound

    # HardSwish backward computation
    scaled_diff = diff * one_third
    adjusted_diff = scaled_diff + half
    weighted_diff = weight_tensor * adjusted_diff
    conditional_result = tl.where(is_within_bounds, weighted_diff, weight_tensor)

    # Apply lower bound condition
    final_result = tl.where(is_below_lower, 0.0, conditional_result)

    # Store the result back
    tl.store(in_out_ptr0 + (x0), final_result, xmask)