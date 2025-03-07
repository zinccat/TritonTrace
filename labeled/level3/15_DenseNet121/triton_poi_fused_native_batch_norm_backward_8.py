# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_batch_norm_backward_8poi_fused_native_batch_norm_backward_8(input_grad_ptr, running_mean_ptr, output_grad_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    channel_index = (block_indices // 3136) % 64
    
    input_grad = tl.load(input_grad_ptr + (index), None)
    running_mean = tl.load(running_mean_ptr + (channel_index), None, eviction_policy='evict_last')
    
    output_grad = input_grad - running_mean
    tl.store(output_grad_ptr + (index), output_grad, None)