# From: 64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_exp_gelu_gelu_backward_leaky_relu_leaky_relu_backward_mul_sub_1poi_fused_exp_gelu_gelu_backward_leaky_relu_leaky_relu_backward_mul_sub_1(
    in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    block_id = xindex // 512
    element_id = xindex
    input_data0 = tl.load(in_ptr0 + (block_id), xmask, eviction_policy='evict_last')
    input_data1 = tl.load(in_ptr1 + (block_id), xmask, eviction_policy='evict_last')
    in_out_data = tl.load(in_out_ptr0 + (element_id), xmask)
    
    zero = 0.0
    leaky_relu_threshold = 0.01
    gelu_coefficient = 0.7071067811865476
    erf_addition = 1.0
    gelu_half = 0.5
    erf_negative_half = -0.5
    exp_coefficient = 0.3989422804014327
    
    is_positive = input_data0 > zero
    leaky_relu_output = tl.where(is_positive, input_data0, input_data0 * leaky_relu_threshold)
    is_leaky_relu_positive = leaky_relu_output > zero
    leaky_relu_output_scaled = leaky_relu_output * leaky_relu_threshold
    scaled_output = tl.where(is_leaky_relu_positive, leaky_relu_output, leaky_relu_output_scaled)
    
    scaled_gelu_input = scaled_output * gelu_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(scaled_gelu_input)
    erf_result_scaled = erf_result + erf_addition
    gelu_approximation = erf_result_scaled * gelu_half
    
    squared_scaled_output = scaled_output * scaled_output
    exp_argument = squared_scaled_output * erf_negative_half
    exp_result = tl.math.exp(exp_argument)
    exp_coefficient_scaled = exp_result * exp_coefficient
    gelu_correction = scaled_output * exp_coefficient_scaled
    gelu_final = gelu_approximation + gelu_correction
    
    gelu_weighted = input_data1 * gelu_final
    gelu_weighted_scaled = gelu_weighted * leaky_relu_threshold
    gelu_weighted_leaky_relu = tl.where(is_leaky_relu_positive, gelu_weighted, gelu_weighted_scaled)
    gelu_weighted_leaky_relu_scaled = gelu_weighted_leaky_relu * leaky_relu_threshold
    final_output = tl.where(is_positive, gelu_weighted_leaky_relu, gelu_weighted_leaky_relu_scaled)
    
    input_out_diff = in_out_data - input_data0
    exp_diff = tl.math.exp(input_out_diff)
    final_result = final_output * exp_diff
    
    tl.store(in_out_ptr0 + (element_id), final_result, xmask)