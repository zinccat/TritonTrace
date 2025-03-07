# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_76poi_fused_cat_76(input_ptr, output_ptr1, output_ptr2, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 62720
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    linear_index = block_indices
    channel_index = linear_index % 6272
    batch_index = linear_index // 6272
    temp_data = tl.load(input_ptr + (linear_index), mask)
    tl.store(output_ptr1 + (channel_index + 257152 * batch_index), temp_data, mask)
    tl.store(output_ptr2 + (channel_index + 263424 * batch_index), temp_data, mask)