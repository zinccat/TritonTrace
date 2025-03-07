# From: 13_DenseNet121TransitionLayer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_4poi_fused_avg_pool2d_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    col_index = block_indices % 112
    row_index = block_indices // 112
    linear_index = block_indices
    
    input_value_0 = tl.load(in_ptr0 + (2 * col_index + 448 * row_index), None, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr0 + (1 + 2 * col_index + 448 * row_index), None, eviction_policy='evict_last')
    input_value_3 = tl.load(in_ptr0 + (224 + 2 * col_index + 448 * row_index), None, eviction_policy='evict_last')
    input_value_5 = tl.load(in_ptr0 + (225 + 2 * col_index + 448 * row_index), None, eviction_policy='evict_last')
    
    sum_01 = input_value_1 + input_value_0
    sum_023 = input_value_3 + sum_01
    sum_0235 = input_value_5 + sum_023
    
    avg_pool_factor = 0.25
    avg_pooled_value = sum_0235 * avg_pool_factor
    
    tl.store(out_ptr0 + (linear_index), avg_pooled_value, None)