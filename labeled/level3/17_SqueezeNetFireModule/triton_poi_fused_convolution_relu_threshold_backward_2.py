# From: 17_SqueezeNetFireModule

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_2poi_fused_convolution_relu_threshold_backward_2(
    input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    channel_index = (block_indices // 50176) % 64
    
    input_value0 = tl.load(input_ptr0 + (index), None)
    input_value1 = tl.load(input_ptr1 + (channel_index), None, eviction_policy='evict_last')
    
    sum_values = input_value0 + input_value1
    zero_tensor = tl.full([1], 0, tl.int32)
    max_value = triton_helpers.maximum(zero_tensor, sum_values)
    
    threshold = 0.0
    result = max_value <= threshold
    
    tl.store(output_ptr0 + (index), result, None)