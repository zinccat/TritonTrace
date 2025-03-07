# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_33poi_fused_cat_33(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 250880
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    linear_index = block_indices
    element_index = linear_index % 25088
    batch_index = linear_index // 25088
    temp_data = tl.load(input_ptr + (linear_index), valid_mask)
    tl.store(output_ptr + (element_index + 401408 * batch_index), temp_data, valid_mask)