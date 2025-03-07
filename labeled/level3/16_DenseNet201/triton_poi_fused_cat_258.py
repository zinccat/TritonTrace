# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_258poi_fused_cat_258(input_ptr, output_ptr0, output_ptr1, output_ptr2, output_ptr3, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 15680
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = global_indices % 1568
    block_indices = global_indices // 1568
    temp_data = tl.load(input_ptr + (global_indices), valid_mask)
    tl.store(output_ptr0 + (local_indices + 58016 * block_indices), temp_data, valid_mask)
    tl.store(output_ptr1 + (local_indices + 59584 * block_indices), temp_data, valid_mask)
    tl.store(output_ptr2 + (local_indices + 61152 * block_indices), temp_data, valid_mask)
    tl.store(output_ptr3 + (local_indices + 62720 * block_indices), temp_data, valid_mask)