# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_225poi_fused_cat_225(input_ptr, output_ptr0, output_ptr1, output_ptr2, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 15680
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 1568
    batch_indices = block_indices // 1568
    temp_data = tl.load(input_ptr + (global_indices), mask)
    tl.store(output_ptr0 + (local_indices + 78400 * batch_indices), temp_data, mask)
    tl.store(output_ptr1 + (local_indices + 79968 * batch_indices), temp_data, mask)
    tl.store(output_ptr2 + (local_indices + 81536 * batch_indices), temp_data, mask)