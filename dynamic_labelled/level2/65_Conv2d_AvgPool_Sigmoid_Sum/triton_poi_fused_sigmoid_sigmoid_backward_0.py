# From: 65_Conv2d_AvgPool_Sigmoid_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_sigmoid_sigmoid_backward_0poi_fused_sigmoid_sigmoid_backward_0(in_out_ptr0, in_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    input_row = index // kernel_size
    input_col = index
    input_value = tl.load(in_ptr0 + (input_row), mask, eviction_policy='evict_last')
    output_value = tl.load(in_out_ptr0 + (input_col), mask, eviction_policy='evict_last')
    sigmoid_output = tl.sigmoid(output_value)
    one_minus_sigmoid = 1.0 - sigmoid_output
    gradient = sigmoid_output * one_minus_sigmoid
    updated_value = input_value * gradient
    tl.store(in_out_ptr0 + (input_col), updated_value, mask)