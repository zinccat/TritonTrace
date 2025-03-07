# From: 5_AlexNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_9poi_fused_max_pool2d_with_indices_9(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 432640
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < num_elements
    x = (index % 256)
    y = ((index // 256) % 13)
    z = ((index // 3328) % 13)
    batch = xindex // 43264
    flat_index = index

    # Load input values
    input_val_0 = tl.load(input_ptr + (x + 512*y + 13824*z + 186624*batch), mask)
    input_val_1 = tl.load(input_ptr + (256 + x + 512*y + 13824*z + 186624*batch), mask)
    input_val_2 = tl.load(input_ptr + (512 + x + 512*y + 13824*z + 186624*batch), mask)
    input_val_3 = tl.load(input_ptr + (6912 + x + 512*y + 13824*z + 186624*batch), mask)
    input_val_4 = tl.load(input_ptr + (7168 + x + 512*y + 13824*z + 186624*batch), mask)
    input_val_5 = tl.load(input_ptr + (7424 + x + 512*y + 13824*z + 186624*batch), mask)
    input_val_6 = tl.load(input_ptr + (13824 + x + 512*y + 13824*z + 186624*batch), mask)
    input_val_7 = tl.load(input_ptr + (14080 + x + 512*y + 13824*z + 186624*batch), mask)
    input_val_8 = tl.load(input_ptr + (14336 + x + 512*y + 13824*z + 186624*batch), mask)

    # Compute maximum values
    max_val_1 = triton_helpers.maximum(input_val_1, input_val_0)
    max_val_2 = triton_helpers.maximum(input_val_2, max_val_1)
    max_val_3 = triton_helpers.maximum(input_val_3, max_val_2)
    max_val_4 = triton_helpers.maximum(input_val_4, max_val_3)
    max_val_5 = triton_helpers.maximum(input_val_5, max_val_4)
    max_val_6 = triton_helpers.maximum(input_val_6, max_val_5)
    max_val_7 = triton_helpers.maximum(input_val_7, max_val_6)
    max_val_8 = triton_helpers.maximum(input_val_8, max_val_7)

    # Determine indices of maximum values
    index_1 = tl.where(input_val_1 > input_val_0, tl.full([1], 1, tl.int8), tl.full([1], 0, tl.int8))
    index_2 = tl.where(input_val_2 > max_val_1, tl.full([1], 2, tl.int8), index_1)
    index_3 = tl.where(input_val_3 > max_val_2, tl.full([1], 3, tl.int8), index_2)
    index_4 = tl.where(input_val_4 > max_val_3, tl.full([1], 4, tl.int8), index_3)
    index_5 = tl.where(input_val_5 > max_val_4, tl.full([1], 5, tl.int8), index_4)
    index_6 = tl.where(input_val_6 > max_val_5, tl.full([1], 6, tl.int8), index_5)
    index_7 = tl.where(input_val_7 > max_val_6, tl.full([1], 7, tl.int8), index_6)
    index_8 = tl.where(input_val_8 > max_val_7, tl.full([1], 8, tl.int8), index_7)

    # Store results
    tl.store(output_ptr_max + (flat_index), max_val_8, mask)
    tl.store(output_ptr_indices + (flat_index), index_8, mask)