# From: 24_LogSoftmax

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__log_softmax_4(input_ptr_max, input_ptr_mean, input_ptr_sum_exp, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    block_index = (block_indices // 16384)
    
    max_value = tl.load(input_ptr_max + (index), None)
    mean_value = tl.load(input_ptr_mean + (block_index), None, eviction_policy='evict_last')
    sum_exp_value = tl.load(input_ptr2 + (block_index), None, eviction_policy='evict_last')
    
    centered_value = max_value - mean_value
    log_sum_exp = tl.math.log(sum_exp_value)
    log_softmax_value = centered_value - log_sum_exp
    
    tl.store(output_ptr + (index), log_softmax_value, None)