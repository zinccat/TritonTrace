# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_219poi_fused_cat_219(input_ptr, output_ptr1, output_ptr2, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 15680
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    linear_index = block_indices
    channel_index = linear_index % 1568
    batch_index = linear_index // 1568
    temp_data = tl.load(input_ptr + (linear_index), valid_mask)
    tl.store(output_ptr1 + (channel_index + 83104 * batch_index), temp_data, valid_mask)
    tl.store(output_ptr2 + (channel_index + 84672 * batch_index), temp_data, valid_mask)