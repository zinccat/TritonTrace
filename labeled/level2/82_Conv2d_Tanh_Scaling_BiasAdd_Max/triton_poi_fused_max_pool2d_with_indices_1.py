# From: 82_Conv2d_Tanh_Scaling_BiasAdd_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x_col = xindex % 15
    x_row = xindex // 15
    x_flat_index = xindex
    x_block_index = xindex // 3600
    x_within_block_index = xindex % 3600
    
    input_val_0 = tl.load(in_ptr0 + ((2 * x_col) + (60 * x_row)), None, eviction_policy='evict_last')
    input_val_1 = tl.load(in_ptr0 + (1 + (2 * x_col) + (60 * x_row)), None, eviction_policy='evict_last')
    input_val_2 = tl.load(in_ptr0 + (30 + (2 * x_col) + (60 * x_row)), None, eviction_policy='evict_last')
    input_val_3 = tl.load(in_ptr0 + (31 + (2 * x_col) + (60 * x_row)), None, eviction_policy='evict_last')
    
    max_val_1 = triton_helpers.maximum(input_val_1, input_val_0)
    max_val_2 = triton_helpers.maximum(input_val_2, max_val_1)
    max_val_final = triton_helpers.maximum(input_val_3, max_val_2)
    
    index_1 = input_val_1 > input_val_0
    index_1_val = tl.full([1], 1, tl.int8)
    index_0_val = tl.full([1], 0, tl.int8)
    max_index_1 = tl.where(index_1, index_1_val, index_0_val)
    
    index_2 = input_val_2 > max_val_1
    index_2_val = tl.full([1], 2, tl.int8)
    max_index_2 = tl.where(index_2, index_2_val, max_index_1)
    
    index_3 = input_val_3 > max_val_2
    index_3_val = tl.full([1], 3, tl.int8)
    max_index_final = tl.where(index_3, index_3_val, max_index_2)
    
    tl.store(out_ptr0 + (x_flat_index), max_val_final, None)
    tl.store(out_ptr1 + (x_within_block_index + (3712 * x_block_index)), max_index_final, None)