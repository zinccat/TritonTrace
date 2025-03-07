# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_26poi_fused_convolution_relu_threshold_backward_26(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 43264
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 256
    input_value0 = tl.load(input_ptr0 + (global_indices), valid_mask)
    input_value1 = tl.load(input_ptr1 + (local_indices), valid_mask, eviction_policy='evict_last')
    sum_values = input_value0 + input_value1
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, sum_values)
    threshold_value = 0.0
    threshold_mask = relu_output <= threshold_value
    tl.store(output_ptr0 + (global_indices), threshold_mask, valid_mask)