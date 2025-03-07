# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_103poi_fused_cat_103(input_ptr, output_ptr0, output_ptr1, output_ptr2, output_ptr3, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 62720
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = global_indices % 6272
    block_indices = global_indices // 6272
    temp_data = tl.load(input_ptr + (global_indices), valid_mask)
    tl.store(output_ptr0 + (local_indices + 106624 * block_indices), temp_data, valid_mask)
    tl.store(output_ptr1 + (local_indices + 112896 * block_indices), temp_data, valid_mask)
    tl.store(output_ptr2 + (local_indices + 119168 * block_indices), temp_data, valid_mask)
    tl.store(output_ptr3 + (local_indices + 125440 * block_indices), temp_data, valid_mask)