# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_gelu_1poi_fused_avg_pool3d_gelu_1(input_ptr, output_ptr1, output_ptr2, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    x0 = (block_indices % 32)
    x1 = ((block_indices // 32) % 32)
    x2 = block_indices // 1024
    x3 = block_indices
    
    input_offset_base = 2 * x0 + 128 * x1 + 8192 * x2
    
    input_val0 = tl.load(input_ptr + input_offset_base, None, eviction_policy='evict_last')
    input_val1 = tl.load(input_ptr + (1 + input_offset_base), None, eviction_policy='evict_last')
    input_val3 = tl.load(input_ptr + (64 + input_offset_base), None, eviction_policy='evict_last')
    input_val5 = tl.load(input_ptr + (65 + input_offset_base), None, eviction_policy='evict_last')
    input_val7 = tl.load(input_ptr + (4096 + input_offset_base), None, eviction_policy='evict_last')
    input_val9 = tl.load(input_ptr + (4097 + input_offset_base), None, eviction_policy='evict_last')
    input_val11 = tl.load(input_ptr + (4160 + input_offset_base), None, eviction_policy='evict_last')
    input_val13 = tl.load(input_ptr + (4161 + input_offset_base), None, eviction_policy='evict_last')
    
    sum1 = input_val1 + input_val0
    sum2 = input_val3 + sum1
    sum3 = input_val5 + sum2
    sum4 = input_val7 + sum3
    sum5 = input_val9 + sum4
    sum6 = input_val11 + sum5
    sum7 = input_val13 + sum6
    
    avg_pool = 0.125 * sum7
    scaled_avg_pool = 0.5 * avg_pool
    erf_input = 0.7071067811865476 * avg_pool
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    final_result = scaled_avg_pool * (erf_result + 1.0)
    
    tl.store(output_ptr1 + (x3), avg_pool, None)
    tl.store(output_ptr2 + (x3), final_result, None)