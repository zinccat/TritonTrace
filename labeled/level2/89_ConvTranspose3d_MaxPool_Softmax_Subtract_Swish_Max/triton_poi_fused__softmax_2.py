# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__softmax_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    global_index = block_indices
    local_index = block_indices % 16384
    batch_index = block_indices // 262144
    
    input_value = tl.load(in_out_ptr0 + (global_index), None)
    max_value = tl.load(in_ptr0 + (local_index + (16384 * batch_index)), None, eviction_policy='evict_last')
    sum_exp_values = tl.load(in_ptr1 + (local_index + (16384 * batch_index)), None, eviction_policy='evict_last')
    
    adjusted_value = input_value - max_value
    exp_value = tl.math.exp(adjusted_value)
    softmax_value = exp_value / sum_exp_values
    
    tl.store(in_out_ptr0 + (global_index), softmax_value, None)