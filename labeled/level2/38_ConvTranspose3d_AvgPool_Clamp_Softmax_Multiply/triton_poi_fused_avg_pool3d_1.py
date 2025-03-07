# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_avg_pool3d_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x_dim = block_indices % 32
    y_dim = (block_indices // 32) % 32
    z_dim = block_indices // 1024
    linear_index = block_indices
    
    input_offset_base = (2 * x_dim) + (128 * y_dim) + (8192 * z_dim)
    
    input_value_0 = tl.load(in_ptr0 + input_offset_base, None, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr0 + (1 + input_offset_base), None, eviction_policy='evict_last')
    input_value_2 = tl.load(in_ptr0 + (64 + input_offset_base), None, eviction_policy='evict_last')
    input_value_3 = tl.load(in_ptr0 + (65 + input_offset_base), None, eviction_policy='evict_last')
    input_value_4 = tl.load(in_ptr0 + (4096 + input_offset_base), None, eviction_policy='evict_last')
    input_value_5 = tl.load(in_ptr0 + (4097 + input_offset_base), None, eviction_policy='evict_last')
    input_value_6 = tl.load(in_ptr0 + (4160 + input_offset_base), None, eviction_policy='evict_last')
    input_value_7 = tl.load(in_ptr0 + (4161 + input_offset_base), None, eviction_policy='evict_last')
    
    sum_01 = input_value_1 + input_value_0
    sum_23 = input_value_2 + sum_01
    sum_45 = input_value_3 + sum_23
    sum_67 = input_value_4 + sum_45
    sum_89 = input_value_5 + sum_67
    sum_1011 = input_value_6 + sum_89
    total_sum = input_value_7 + sum_1011
    
    avg_pool_factor = 0.125
    avg_pooled_value = total_sum * avg_pool_factor
    
    tl.store(out_ptr0 + linear_index, avg_pooled_value, None)