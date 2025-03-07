# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_backward_relu_threshold_backward_6poi_fused_convolution_backward_relu_threshold_backward_6(in_out_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    indices = block_indices
    output_values = tl.load(in_out_ptr + (indices), mask)
    input_values = tl.load(input_ptr + (indices), mask)
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, output_values)
    threshold_value = 0.0
    threshold_mask = relu_output <= threshold_value
    thresholded_values = tl.where(threshold_mask, threshold_value, input_values)
    tl.store(in_out_ptr + (indices), thresholded_values, mask)