# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_avg_pool3d_4(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    x_dim = block_indices % 31
    y_dim = (block_indices // 31) % 31
    z_dim = (block_indices // 961) % 31
    batch_dim = block_indices // 29791
    linear_index = block_indices % 29791
    
    load_offset = lambda x, y, z, b: (2*x) + (126*y) + (8000*z) + (252000*b)
    
    value0 = tl.load(input_ptr + load_offset(x_dim, y_dim, z_dim, batch_dim), None, eviction_policy='evict_last')
    value1 = tl.load(input_ptr + (1 + load_offset(x_dim, y_dim, z_dim, batch_dim)), None, eviction_policy='evict_last')
    value3 = tl.load(input_ptr + (63 + load_offset(x_dim, y_dim, z_dim, batch_dim)), None, eviction_policy='evict_last')
    value5 = tl.load(input_ptr + (64 + load_offset(x_dim, y_dim, z_dim, batch_dim)), None, eviction_policy='evict_last')
    value7 = tl.load(input_ptr + (4000 + load_offset(x_dim, y_dim, z_dim, batch_dim)), None, eviction_policy='evict_last')
    value9 = tl.load(input_ptr + (4001 + load_offset(x_dim, y_dim, z_dim, batch_dim)), None, eviction_policy='evict_last')
    value11 = tl.load(input_ptr + (4063 + load_offset(x_dim, y_dim, z_dim, batch_dim)), None, eviction_policy='evict_last')
    value13 = tl.load(input_ptr + (4064 + load_offset(x_dim, y_dim, z_dim, batch_dim)), None, eviction_policy='evict_last')
    
    sum1 = value1 + value0
    sum3 = value3 + sum1
    sum5 = value5 + sum3
    sum7 = value7 + sum5
    sum9 = value9 + sum7
    sum11 = value11 + sum9
    sum13 = value13 + sum11
    
    avg_value = sum13 * 0.125
    tl.store(output_ptr + (linear_index + (29792 * batch_dim)), avg_value, None)