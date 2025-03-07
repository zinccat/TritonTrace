# From: 3_DeepNarrowMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_relu_0poi_fused_addmm_relu_0(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 50
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    output_values = tl.load(output_ptr + (indices), valid_mask)
    input_values = tl.load(input_ptr + (indices), valid_mask)
    summed_values = output_values + input_values
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_values = triton_helpers.maximum(zero_tensor, summed_values)
    tl.store(output_ptr + (indices), relu_values, valid_mask)