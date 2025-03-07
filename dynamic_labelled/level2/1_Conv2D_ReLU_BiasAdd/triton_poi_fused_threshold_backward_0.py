# From: 1_Conv2D_ReLU_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_threshold_backward_0poi_fused_threshold_backward_0(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    indices = block_indices
    input_data0 = tl.load(input_ptr0 + (indices), mask).to(tl.int1)
    input_data1 = tl.load(input_ptr1 + (indices), mask)
    threshold_value = 0.0
    result_data = tl.where(input_data0, threshold_value, input_data1)
    tl.store(output_ptr0 + (indices), result_data, mask)