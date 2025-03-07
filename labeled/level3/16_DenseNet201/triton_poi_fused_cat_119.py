# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_119poi_fused_cat_119(input_ptr, output_ptr1, output_ptr2, total_elements, BLOCK_SIZE : tl.constexpr):
    total_elements = 62720
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements
    linear_index = block_indices
    element_index_within_block = linear_index % 6272
    block_index = linear_index // 6272
    loaded_data = tl.load(input_ptr + (linear_index), valid_mask)
    tl.store(output_ptr1 + (element_index_within_block + 288512 * block_index), loaded_data, valid_mask)
    tl.store(output_ptr2 + (element_index_within_block + 294784 * block_index), loaded_data, valid_mask)