# From: 1_Conv2D_ReLU_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_threshold_backward_0(input_grad_ptr, input_data_ptr, output_grad_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    grad_mask = tl.load(input_grad_ptr + (indices), valid_mask).to(tl.int1)
    input_data = tl.load(input_data_ptr + (indices), valid_mask)
    zero_value = 0.0
    output_data = tl.where(grad_mask, zero_value, input_data)
    tl.store(output_grad_ptr + (indices), output_data, valid_mask)