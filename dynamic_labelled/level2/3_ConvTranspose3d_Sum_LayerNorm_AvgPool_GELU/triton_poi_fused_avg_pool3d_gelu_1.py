# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_gelu_1(input_ptr, output_ptr1, output_ptr2, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    x_mod_32 = block_index % 32
    x_div_32_mod_32 = (block_index // 32) % 32
    x_div_1024 = block_index // 1024
    linear_index = block_index
    
    input_offset_base = 2 * x_mod_32 + 128 * x_div_32_mod_32 + 8192 * x_div_1024
    
    input_value_0 = tl.load(input_ptr + input_offset_base, None, eviction_policy='evict_last')
    input_value_1 = tl.load(input_ptr + (1 + input_offset_base), None, eviction_policy='evict_last')
    input_value_3 = tl.load(input_ptr + (64 + input_offset_base), None, eviction_policy='evict_last')
    input_value_5 = tl.load(input_ptr + (65 + input_offset_base), None, eviction_policy='evict_last')
    input_value_7 = tl.load(input_ptr + (4096 + input_offset_base), None, eviction_policy='evict_last')
    input_value_9 = tl.load(input_ptr + (4097 + input_offset_base), None, eviction_policy='evict_last')
    input_value_11 = tl.load(input_ptr + (4160 + input_offset_base), None, eviction_policy='evict_last')
    input_value_13 = tl.load(input_ptr + (4161 + input_offset_base), None, eviction_policy='evict_last')
    
    sum_2 = input_value_1 + input_value_0
    sum_4 = input_value_3 + sum_2
    sum_6 = input_value_5 + sum_4
    sum_8 = input_value_7 + sum_6
    sum_10 = input_value_9 + sum_8
    sum_12 = input_value_11 + sum_10
    sum_14 = input_value_13 + sum_12
    
    avg_pool_factor = 0.125
    avg_pooled_value = sum_14 * avg_pool_factor
    
    gelu_factor_1 = 0.5
    gelu_factor_2 = 0.7071067811865476
    erf_input = avg_pooled_value * gelu_factor_2
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    
    gelu_result = avg_pooled_value * gelu_factor_1 * (erf_result + 1.0)
    
    tl.store(output_ptr1 + linear_index, avg_pooled_value, None)
    tl.store(output_ptr2 + linear_index, gelu_result, None)