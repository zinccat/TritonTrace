# From: 64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_exp_gelu_gelu_backward_leaky_relu_leaky_relu_backward_mul_sub_1(
    in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    block_id = xindex // 512
    element_index = xindex
    input_value = tl.load(in_ptr0 + (block_id), xmask, eviction_policy='evict_last')
    grad_output_value = tl.load(in_ptr1 + (block_id), xmask, eviction_policy='evict_last')
    in_out_value = tl.load(in_out_ptr0 + (element_index), xmask)
    
    leaky_relu_threshold = 0.0
    is_positive = input_value > leaky_relu_threshold
    leaky_slope = 0.01
    scaled_input = input_value * leaky_slope
    leaky_relu_output = tl.where(is_positive, input_value, scaled_input)
    
    is_positive_leaky = leaky_relu_output > leaky_relu_threshold
    scaled_leaky_output = leaky_relu_output * leaky_slope
    final_leaky_output = tl.where(is_positive_leaky, leaky_relu_output, scaled_leaky_output)
    
    erf_coefficient = 0.7071067811865476
    scaled_for_erf = final_leaky_output * erf_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(scaled_for_erf)
    
    erf_offset = 1.0
    erf_sum = erf_result + erf_offset
    erf_half = 0.5
    erf_half_sum = erf_sum * erf_half
    
    squared_output = final_leaky_output * final_leaky_output
    exp_coefficient = -0.5
    exp_argument = squared_output * exp_coefficient
    exp_result = tl.math.exp(exp_argument)
    
    gaussian_coefficient = 0.3989422804014327
    gaussian_result = exp_result * gaussian_coefficient
    gaussian_term = final_leaky_output * gaussian_result
    gelu_result = erf_half_sum + gaussian_term
    
    grad_output_scaled = grad_output_value * gelu_result
    gelu_leaky_slope = 0.01
    scaled_grad_output = grad_output_scaled * gelu_leaky_slope
    final_grad_output = tl.where(is_positive_leaky, grad_output_scaled, scaled_grad_output)
    
    final_scaled_grad_output = final_grad_output * gelu_leaky_slope
    final_leaky_grad_output = tl.where(is_positive, final_grad_output, final_scaled_grad_output)
    
    input_diff = in_out_value - input_value
    exp_diff = tl.math.exp(input_diff)
    final_result = final_leaky_grad_output * exp_diff
    
    tl.store(in_out_ptr0 + (element_index), final_result, xmask)