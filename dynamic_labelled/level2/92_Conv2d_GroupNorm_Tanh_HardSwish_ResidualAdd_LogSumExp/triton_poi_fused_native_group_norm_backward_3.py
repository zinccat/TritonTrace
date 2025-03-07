# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_3(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, output_grad_ptr, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x2 = x_index
    x0 = (x_index % 8)
    
    input_grad_0 = tl.load(input_grad_ptr + (2 * x2), x_mask, eviction_policy='evict_last')
    gamma_0 = tl.load(input_ptr + (2 * x0), x_mask, eviction_policy='evict_last')
    input_grad_1 = tl.load(input_grad_ptr + (1 + 2 * x2), x_mask, eviction_policy='evict_last')
    gamma_1 = tl.load(input_ptr + (1 + 2 * x0), x_mask, eviction_policy='evict_last')
    mean = tl.load(mean_ptr + (x2), x_mask)
    gamma_grad_0 = tl.load(gamma_ptr + (2 * x2), x_mask, eviction_policy='evict_last')
    gamma_grad_1 = tl.load(gamma_ptr + (1 + 2 * x2), x_mask, eviction_policy='evict_last')
    variance = tl.load(variance_ptr + (x2), x_mask)
    
    gamma_input_grad_0 = input_grad_0 * gamma_0
    gamma_input_grad_1 = input_grad_1 * gamma_1
    sum_gamma_input_grad = gamma_input_grad_0 + gamma_input_grad_1
    normalized_grad = sum_gamma_input_grad * mean
    
    gamma_grad_0_input_grad = gamma_grad_0 * gamma_0
    gamma_grad_1_input_grad = gamma_grad_1 * gamma_1
    sum_gamma_grad_input_grad = gamma_grad_0_input_grad + gamma_grad_1_input_grad
    diff_grad = normalized_grad - sum_gamma_grad_input_grad
    
    variance_scaled_diff = diff_grad * variance
    variance_scaled_diff_cubed = variance_scaled_diff * variance_scaled_diff * variance_scaled_diff
    
    neg_two = -2.0
    kernel_size_float = kernel_size.to(tl.float32)
    neg_two_plus_kernel_size = neg_two + kernel_size_float
    two = 2.0
    power_result = tl.extra.cuda.libdevice.pow(neg_two_plus_kernel_size, two)
    two_times_power_result = two * power_result
    double_two_times_power_result = two_times_power_result.to(tl.float64)
    
    one = tl.full([1], 1.0, tl.float64)
    reciprocal_double_power = one / double_two_times_power_result
    reciprocal_float = reciprocal_double_power.to(tl.float32)
    
    scaled_variance_grad = variance_scaled_diff_cubed * reciprocal_float
    neg_scaled_variance_grad = -scaled_variance_grad
    scaled_mean_grad = neg_scaled_variance_grad * mean
    
    mean_scaled_grad = sum_gamma_input_grad * variance
    scaled_mean_grad_reciprocal = mean_scaled_grad * reciprocal_float
    final_mean_grad = scaled_mean_grad - scaled_mean_grad_reciprocal
    
    tl.store(output_grad_ptr + (x2), scaled_variance_grad, x_mask)
    tl.store(output_ptr + (x2), final_mean_grad, x_mask)