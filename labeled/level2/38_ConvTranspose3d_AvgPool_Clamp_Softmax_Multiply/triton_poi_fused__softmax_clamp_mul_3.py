# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused__softmax_clamp_mul_3(input_ptr0, input_ptr1, input_ptr2, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    global_index = block_indices
    local_index = block_indices % 16384
    batch_index = block_indices // 262144
    
    input_value0 = tl.load(input_ptr0 + (global_index), None)
    input_value1 = tl.load(input_ptr1 + (local_index + (16384 * batch_index)), None, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (local_index + (16384 * batch_index)), None, eviction_policy='evict_last')
    
    clamp_min = 0.0
    max_value = triton_helpers.maximum(input_value0, clamp_min)
    clamp_max = 1.0
    clamped_value = triton_helpers.minimum(max_value, clamp_max)
    
    subtracted_value = clamped_value - input_value1
    exp_value = tl.math.exp(subtracted_value)
    softmax_output = exp_value / input_value2
    
    scaled_output = softmax_output * 2.0
    tl.store(output_ptr0 + (global_index), scaled_output, None)