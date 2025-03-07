# From: 36_RMSNorm_

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_div_mean_pow_sqrt_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    global_index = block_indices
    local_index = block_indices % 65536
    block_index = block_indices // 4194304
    
    input_value0 = tl.load(in_ptr0 + (global_index), None)
    input_value1 = tl.load(in_ptr1 + (local_index + (65536 * block_index)), None, eviction_policy='evict_last')
    
    divisor = 64.0
    epsilon = 1e-05
    
    normalized_value = input_value1 / divisor
    adjusted_value = normalized_value + epsilon
    sqrt_value = tl.extra.cuda.libdevice.sqrt(adjusted_value)
    
    result_value = input_value0 / sqrt_value
    tl.store(out_ptr0 + (global_index), result_value, None)