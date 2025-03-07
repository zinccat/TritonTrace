# From: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_1poi_fused_div_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    block_id = xindex // 512
    local_id = xindex
    input0 = tl.load(in_ptr0 + (block_id), xmask, eviction_policy='evict_last')
    input1 = tl.load(in_ptr1 + (block_id), xmask, eviction_policy='evict_last')
    abs_input1 = tl.math.abs(input1)
    inf_value = float("inf")
    is_inf = abs_input1 == inf_value
    zero_value = 0.0
    safe_input1 = tl.where(is_inf, zero_value, input1)
    input1_shifted = input1 - safe_input1
    exp_shifted = tl.math.exp(input1_shifted)
    log_exp = tl.math.log(exp_shifted)
    log_sum_exp = log_exp + safe_input1
    sqrt_2_over_2 = 0.7071067811865476
    erf_input = log_sum_exp * sqrt_2_over_2
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    one_value = 1.0
    erf_plus_one = erf_result + one_value
    half_value = 0.5
    erf_half = erf_plus_one * half_value
    log_sum_exp_squared = log_sum_exp * log_sum_exp
    neg_half = -0.5
    exp_neg_half_squared = tl.math.exp(log_sum_exp_squared * neg_half)
    sqrt_2_pi = 0.3989422804014327
    gaussian = exp_neg_half_squared * sqrt_2_pi
    gaussian_scaled = log_sum_exp * gaussian
    gelu_result = erf_half + gaussian_scaled
    input0_scaled = input0 * gelu_result
    input1_log_sum_exp_diff = input1 - log_sum_exp
    exp_diff = tl.math.exp(input1_log_sum_exp_diff)
    scaled_result = input0_scaled * exp_diff
    scale_factor = 0.001953125
    final_result = scaled_result * scale_factor
    tl.store(out_ptr0 + (local_id), final_result, xmask)