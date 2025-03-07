# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_13poi_fused_max_pool2d_with_indices_13(input_ptr, output_ptr_max, output_ptr_indices, total_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    x_col = (block_indices % 64)
    x_row = ((block_indices // 64) % 112)
    x_depth = block_indices // 7168
    linear_index = block_indices
    
    input_offset_0 = x_col + 128 * x_row + 28672 * x_depth
    input_offset_1 = 64 + x_col + 128 * x_row + 28672 * x_depth
    input_offset_2 = 14336 + x_col + 128 * x_row + 28672 * x_depth
    input_offset_3 = 14400 + x_col + 128 * x_row + 28672 * x_depth
    
    value_0 = tl.load(input_ptr + input_offset_0, None)
    value_1 = tl.load(input_ptr + input_offset_1, None)
    value_2 = tl.load(input_ptr + input_offset_2, None)
    value_3 = tl.load(input_ptr + input_offset_3, None)
    
    max_01 = triton_helpers.maximum(value_1, value_0)
    max_23 = triton_helpers.maximum(value_2, max_01)
    max_45 = triton_helpers.maximum(value_3, max_23)
    
    index_01 = tl.full([1], 1, tl.int8)
    index_00 = tl.full([1], 0, tl.int8)
    index_01_or_00 = tl.where(value_1 > value_0, index_01, index_00)
    
    index_23 = tl.full([1], 2, tl.int8)
    index_23_or_01_00 = tl.where(value_2 > max_01, index_23, index_01_or_00)
    
    index_45 = tl.full([1], 3, tl.int8)
    final_index = tl.where(value_3 > max_23, index_45, index_23_or_01_00)
    
    tl.store(output_ptr_max + linear_index, max_45, None)
    tl.store(output_ptr_indices + linear_index, final_index, None)