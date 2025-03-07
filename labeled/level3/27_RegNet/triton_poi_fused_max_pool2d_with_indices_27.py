# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_27poi_fused_max_pool2d_with_indices_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    col_index = block_indices % 256
    row_index = (block_indices // 256) % 28
    channel_index = block_indices // 7168
    linear_index = block_indices
    
    input_value_0 = tl.load(in_ptr0 + (col_index + 512 * row_index + 28672 * channel_index), None)
    input_value_1 = tl.load(in_ptr0 + (256 + col_index + 512 * row_index + 28672 * channel_index), None)
    input_value_7 = tl.load(in_ptr0 + (14336 + col_index + 512 * row_index + 28672 * channel_index), None)
    input_value_12 = tl.load(in_ptr0 + (14592 + col_index + 512 * row_index + 28672 * channel_index), None)
    
    is_value_1_greater = input_value_1 > input_value_0
    index_1 = tl.full([1], 1, tl.int8)
    index_0 = tl.full([1], 0, tl.int8)
    max_index_01 = tl.where(is_value_1_greater, index_1, index_0)
    max_value_01 = triton_helpers.maximum(input_value_1, input_value_0)
    
    is_value_7_greater = input_value_7 > max_value_01
    index_7 = tl.full([1], 2, tl.int8)
    max_index_017 = tl.where(is_value_7_greater, index_7, max_index_01)
    max_value_017 = triton_helpers.maximum(input_value_7, max_value_01)
    
    is_value_12_greater = input_value_12 > max_value_017
    index_12 = tl.full([1], 3, tl.int8)
    max_index_01172 = tl.where(is_value_12_greater, index_12, max_index_017)
    max_value_01172 = triton_helpers.maximum(input_value_12, max_value_017)
    
    tl.store(out_ptr0 + (linear_index), max_index_01172, None)