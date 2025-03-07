# From: 12_VGG19

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_10poi_fused_max_pool2d_with_indices_10(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    x_col = (block_indices % 64)
    x_row = ((block_indices // 64) % 112)
    x_channel = block_indices // 7168
    linear_index = block_indices
    
    input_offset = x_col + 128 * x_row + 28672 * x_channel
    
    input_value_0 = tl.load(input_ptr + input_offset, None)
    input_value_1 = tl.load(input_ptr + (64 + input_offset), None)
    input_value_3 = tl.load(input_ptr + (14336 + input_offset), None)
    input_value_5 = tl.load(input_ptr + (14400 + input_offset), None)
    
    max_val_1_0 = triton_helpers.maximum(input_value_1, input_value_0)
    max_val_3_2 = triton_helpers.maximum(input_value_3, max_val_1_0)
    max_val_5_4 = triton_helpers.maximum(input_value_5, max_val_3_2)
    
    index_1_gt_0 = input_value_1 > input_value_0
    index_1 = tl.full([1], 1, tl.int8)
    index_0 = tl.full([1], 0, tl.int8)
    index_1_0 = tl.where(index_1_gt_0, index_1, index_0)
    
    index_3_gt_2 = input_value_3 > max_val_1_0
    index_3 = tl.full([1], 2, tl.int8)
    index_3_2 = tl.where(index_3_gt_2, index_3, index_1_0)
    
    index_5_gt_4 = input_value_5 > max_val_3_2
    index_5 = tl.full([1], 3, tl.int8)
    max_index = tl.where(index_5_gt_4, index_5, index_3_2)
    
    tl.store(output_ptr_max + linear_index, max_val_5_4, None)
    tl.store(output_ptr_indices + linear_index, max_index, None)