# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_avg_pool3d_gelu_1(input_ptr, output_ptr_activation, output_ptr_gelu, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    x0 = block_indices % 32
    x1 = (block_indices // 32) % 32
    x2 = block_indices // 1024
    x3 = block_indices
    
    input_offset_base = (2 * x0) + (128 * x1) + (8192 * x2)
    
    input_val_0 = tl.load(input_ptr + input_offset_base, None, eviction_policy='evict_last')
    input_val_1 = tl.load(input_ptr + (1 + input_offset_base), None, eviction_policy='evict_last')
    input_val_64 = tl.load(input_ptr + (64 + input_offset_base), None, eviction_policy='evict_last')
    input_val_65 = tl.load(input_ptr + (65 + input_offset_base), None, eviction_policy='evict_last')
    input_val_4096 = tl.load(input_ptr + (4096 + input_offset_base), None, eviction_policy='evict_last')
    input_val_4097 = tl.load(input_ptr + (4097 + input_offset_base), None, eviction_policy='evict_last')
    input_val_4160 = tl.load(input_ptr + (4160 + input_offset_base), None, eviction_policy='evict_last')
    input_val_4161 = tl.load(input_ptr + (4161 + input_offset_base), None, eviction_policy='evict_last')
    
    sum_0_1 = input_val_1 + input_val_0
    sum_2_3 = input_val_64 + sum_0_1
    sum_4_5 = input_val_65 + sum_2_3
    sum_6_7 = input_val_4096 + sum_4_5
    sum_8_9 = input_val_4097 + sum_6_7
    sum_10_11 = input_val_4160 + sum_8_9
    sum_12_13 = input_val_4161 + sum_10_11
    
    avg_pool_result = sum_12_13 * 0.125
    scaled_avg_pool = avg_pool_result * 0.5
    sqrt_2_over_2 = avg_pool_result * 0.7071067811865476
    erf_result = tl.extra.cuda.libdevice.erf(sqrt_2_over_2)
    gelu_result = scaled_avg_pool * (erf_result + 1.0)
    
    tl.store(output_ptr_activation + x3, avg_pool_result, None)
    tl.store(output_ptr_gelu + x3, gelu_result, None)