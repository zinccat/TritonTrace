# From: 69_Conv2d_HardSwish_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_relu_threshold_backward_1poi_fused_hardswish_relu_threshold_backward_1(
    input_ptr, output_ptr1, output_ptr2, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    indices = block_indices

    input_values = tl.load(input_ptr + (indices), mask)
    bias = 3.0
    biased_input = input_values + bias
    lower_bound = 0.0
    clamped_input = triton_helpers.maximum(biased_input, lower_bound)
    upper_bound = 6.0
    capped_input = triton_helpers.minimum(clamped_input, upper_bound)
    scaled_input = input_values * capped_input
    scale_factor = 0.16666666666666666
    scaled_output = scaled_input * scale_factor
    zero_tensor = tl.full([1], 0, tl.int32)
    max_output = triton_helpers.maximum(zero_tensor, scaled_output)
    threshold_check = max_output <= lower_bound

    tl.store(output_ptr1 + (indices), max_output, mask)
    tl.store(output_ptr2 + (indices), threshold_check, mask)