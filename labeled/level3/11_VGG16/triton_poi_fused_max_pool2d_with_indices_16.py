# From: 11_VGG16

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_16poi_fused_max_pool2d_with_indices_16(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    channel_index = block_indices % 512
    height_index = (block_indices // 512) % 14
    batch_index = block_indices // 7168
    linear_index = block_indices
    
    input_offset_0 = channel_index + 1024 * height_index + 28672 * batch_index
    input_offset_1 = 512 + input_offset_0
    input_offset_2 = 14336 + input_offset_0
    input_offset_3 = 14848 + input_offset_0
    
    value_0 = tl.load(input_ptr + input_offset_0, None)
    value_1 = tl.load(input_ptr + input_offset_1, None)
    value_2 = tl.load(input_ptr + input_offset_2, None)
    value_3 = tl.load(input_ptr + input_offset_3, None)
    
    max_01 = triton_helpers.maximum(value_1, value_0)
    max_23 = triton_helpers.maximum(value_2, max_01)
    max_45 = triton_helpers.maximum(value_3, max_23)
    
    index_01 = tl.where(value_1 > value_0, tl.full([1], 1, tl.int8), tl.full([1], 0, tl.int8))
    index_23 = tl.where(value_2 > max_01, tl.full([1], 2, tl.int8), index_01)
    index_45 = tl.where(value_3 > max_23, tl.full([1], 3, tl.int8), index_23)
    
    tl.store(output_ptr_max + linear_index, max_45, None)
    tl.store(output_ptr_indices + linear_index, index_45, None)