# From: 69_Conv2d_HardSwish_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_relu_threshold_backward_1(
    input_ptr, output_ptr1, output_ptr2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    index = indices

    input_value = tl.load(input_ptr + (index), mask)
    bias = 3.0
    biased_input = input_value + bias
    lower_bound = 0.0
    clamped_input = triton_helpers.maximum(biased_input, lower_bound)
    upper_bound = 6.0
    capped_input = triton_helpers.minimum(clamped_input, upper_bound)
    scaled_input = input_value * capped_input
    scale_factor = 0.16666666666666666
    scaled_output = scaled_input * scale_factor
    zero_tensor = tl.full([1], 0, tl.int32)
    max_output = triton_helpers.maximum(zero_tensor, scaled_output)
    threshold_check = max_output <= lower_bound

    tl.store(output_ptr1 + (index), max_output, mask)
    tl.store(output_ptr2 + (index), threshold_check, mask)