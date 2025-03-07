# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_72poi_fused_cat_72(input_ptr, output_ptr0, output_ptr1, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 62720
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    linear_index = block_indices
    mod_index = block_indices % 6272
    div_index = block_indices // 6272
    temp_data = tl.load(input_ptr + (linear_index), mask)
    tl.store(output_ptr0 + (mod_index + 206976 * div_index), temp_data, mask)
    tl.store(output_ptr1 + (mod_index + 213248 * div_index), temp_data, mask)