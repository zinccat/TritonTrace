# From: 28_VisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_relu_threshold_backward_9poi_fused_relu_threshold_backward_9(
    in_out_ptr, input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    global_indices = block_indices
    local_indices = block_indices % 2048
    
    in_out_values = tl.load(in_out_ptr + (global_indices), None)
    input_values = tl.load(input_ptr + (local_indices), None, eviction_policy='evict_last')
    
    sum_values = in_out_values + input_values
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_values = triton_helpers.maximum(zero_tensor, sum_values)
    
    threshold_value = 0.0
    threshold_mask = relu_values <= threshold_value
    
    tl.store(in_out_ptr + (global_indices), relu_values, None)
    tl.store(output_ptr + (global_indices), threshold_mask, None)