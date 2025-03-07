# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_177poi_fused_cat_177(input_ptr, output_ptr0, output_ptr1, output_ptr2, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 15680
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_index = block_indices
    local_index = global_index % 1568
    batch_index = global_index // 1568
    temp_data = tl.load(input_ptr + (global_index), valid_mask)
    tl.store(output_ptr0 + (local_index + 47040 * batch_index), temp_data, valid_mask)
    tl.store(output_ptr1 + (local_index + 48608 * batch_index), temp_data, valid_mask)
    tl.store(output_ptr2 + (local_index + 50176 * batch_index), temp_data, valid_mask)