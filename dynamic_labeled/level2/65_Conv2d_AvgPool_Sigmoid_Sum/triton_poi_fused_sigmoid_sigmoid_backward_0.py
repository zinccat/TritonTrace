# From: 65_Conv2d_AvgPool_Sigmoid_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_sigmoid_sigmoid_backward_0(in_out_ptr0, in_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    input_index = index // kernel_size
    output_index = index
    input_value = tl.load(in_ptr0 + (input_index), mask, eviction_policy='evict_last')
    output_value = tl.load(in_out_ptr0 + (output_index), mask, eviction_policy='evict_last')
    sigmoid_output = tl.sigmoid(output_value)
    one = 1.0
    one_minus_sigmoid = one - sigmoid_output
    gradient = sigmoid_output * one_minus_sigmoid
    updated_value = input_value * gradient
    tl.store(in_out_ptr0 + (output_index), updated_value, mask)