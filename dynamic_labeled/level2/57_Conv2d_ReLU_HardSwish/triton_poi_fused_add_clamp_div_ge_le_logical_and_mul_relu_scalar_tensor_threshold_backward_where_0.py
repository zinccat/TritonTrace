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
    zero_int32 = tl.full([1], 0, tl.int32)
    zero_float = 0.0
    three_float = 3.0
    one_sixth_float = 0.16666666666666666
    one_float = 1.0

    # Compute maximum with zero
    max_with_zero = triton_helpers.maximum(zero_int32, input_tensor)

    # Check if max_with_zero is less than or equal to zero
    is_less_equal_zero = max_with_zero <= zero_float

    # Compute scaled max_with_zero
    scaled_max = max_with_zero + three_float
    scaled_max_times_one_sixth = scaled_max * one_sixth_float

    # Clamp scaled_max_times_one_sixth between zero and one
    clamped_value = triton_helpers.minimum(
        triton_helpers.maximum(scaled_max_times_one_sixth, zero_float), one_float
    )

    # Element-wise multiplication
    weighted_clamped_value = weight_tensor * clamped_value

    # Logical conditions
    is_scaled_max_ge_zero = scaled_max_times_one_sixth >= zero_float
    is_scaled_max_le_one = scaled_max_times_one_sixth <= one_float
    is_within_bounds = is_scaled_max_ge_zero & is_scaled_max_le_one

    # Compute alternative value
    weighted_input_tensor = weight_tensor * max_with_zero
    alternative_value = tl.where(is_within_bounds, weighted_input_tensor, zero_float)

    # Compute final value
    scaled_alternative_value = alternative_value * one_sixth_float
    final_value = weighted_clamped_value + scaled_alternative_value

    # Store result
    result = tl.where(is_less_equal_zero, zero_float, final_value)
    tl.store(in_out_ptr0 + (x0), result, xmask)