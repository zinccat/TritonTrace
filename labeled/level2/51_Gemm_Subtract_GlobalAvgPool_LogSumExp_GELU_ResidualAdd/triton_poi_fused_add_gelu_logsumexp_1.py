# From: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_gelu_logsumexp_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    block_id = xindex // 1024
    element_index = xindex
    
    input_value0 = tl.load(in_ptr0 + (block_id), None, eviction_policy='evict_last')
    input_value1 = tl.load(in_ptr1 + (element_index), None)
    
    abs_input_value0 = tl.math.abs(input_value0)
    infinity = float("inf")
    is_infinity = abs_input_value0 == infinity
    zero = 0.0
    adjusted_input_value0 = tl.where(is_infinity, zero, input_value0)
    
    shifted_input_value0 = input_value0 - adjusted_input_value0
    exp_shifted = tl.math.exp(shifted_input_value0)
    log_exp_shifted = tl.math.log(exp_shifted)
    
    logsumexp_result = log_exp_shifted + adjusted_input_value0
    half = 0.5
    scaled_logsumexp = logsumexp_result * half
    
    erf_coefficient = 0.7071067811865476
    erf_input = logsumexp_result * erf_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    
    one = 1.0
    erf_adjusted = erf_result + one
    gelu_result = scaled_logsumexp * erf_adjusted
    
    fused_result = gelu_result + input_value1
    tl.store(out_ptr0 + (element_index), fused_result, None)