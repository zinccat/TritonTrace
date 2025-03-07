# From: 5_AlexNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_relu_13poi_fused_addmm_relu_13(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    global_indices = block_indices
    local_indices = block_indices % 4096
    output_values = tl.load(output_ptr + (global_indices), None)
    input_values = tl.load(input_ptr + (local_indices), None, eviction_policy='evict_last')
    result_values = output_values + input_values
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_values = triton_helpers.maximum(zero_tensor, result_values)
    tl.store(output_ptr + (global_indices), relu_values, None)