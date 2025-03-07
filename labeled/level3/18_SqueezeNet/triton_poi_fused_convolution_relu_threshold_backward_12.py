# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_12poi_fused_convolution_relu_threshold_backward_12(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 373248
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 128
    input_data0 = tl.load(input_ptr0 + (global_indices), mask)
    input_data1 = tl.load(input_ptr1 + (local_indices), mask, eviction_policy='evict_last')
    sum_data = input_data0 + input_data1
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, sum_data)
    threshold_value = 0.0
    threshold_mask = relu_output <= threshold_value
    tl.store(output_ptr0 + (global_indices), threshold_mask, mask)