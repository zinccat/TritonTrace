# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_58poi_fused_cat_58(input_ptr, output_ptr0, output_ptr1, output_ptr2, output_ptr3, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 250880
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = global_indices % 25088
    batch_indices = global_indices // 25088
    temp_data = tl.load(input_ptr + (global_indices), valid_mask)
    tl.store(output_ptr0 + (local_indices + 326144 * batch_indices), temp_data, valid_mask)
    tl.store(output_ptr1 + (local_indices + 351232 * batch_indices), temp_data, valid_mask)
    tl.store(output_ptr2 + (local_indices + 376320 * batch_indices), temp_data, valid_mask)
    tl.store(output_ptr3 + (local_indices + 401408 * batch_indices), temp_data, valid_mask)