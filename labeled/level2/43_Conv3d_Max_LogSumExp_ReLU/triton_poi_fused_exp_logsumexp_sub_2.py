# From: 43_Conv3d_Max_LogSumExp_ReLU

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_exp_logsumexp_sub_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    global_index = block_indices
    local_index = block_indices % 2048
    block_index = block_indices // 32768
    
    current_value = tl.load(in_out_ptr0 + (global_index), None)
    input_value0 = tl.load(in_ptr0 + (local_index + (2048 * block_index)), None, eviction_policy='evict_last')
    input_value1 = tl.load(in_ptr1 + (local_index + (2048 * block_index)), None, eviction_policy='evict_last')
    
    log_input_value0 = tl.math.log(input_value0)
    abs_input_value1 = tl.math.abs(input_value1)
    
    inf_value = float("inf")
    is_inf = abs_input_value1 == inf_value
    zero_value = 0.0
    adjusted_input_value1 = tl.where(is_inf, zero_value, input_value1)
    
    log_sum_exp = log_input_value0 + adjusted_input_value1
    difference = current_value - log_sum_exp
    
    exp_difference = tl.math.exp(difference)
    tl.store(in_out_ptr0 + (global_index), exp_difference, None)