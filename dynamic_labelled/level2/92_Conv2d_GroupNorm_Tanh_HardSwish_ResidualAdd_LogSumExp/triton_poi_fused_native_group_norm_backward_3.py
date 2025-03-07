# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_3poi_fused_native_group_norm_backward_3(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, output_grad_ptr, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x2 = x_index
    x0 = (x_index % 8)
    
    grad_input_0 = tl.load(input_grad_ptr + (2 * x2), x_mask, eviction_policy='evict_last')
    gamma_0 = tl.load(input_ptr + (2 * x0), x_mask, eviction_policy='evict_last')
    grad_input_1 = tl.load(input_grad_ptr + (1 + 2 * x2), x_mask, eviction_policy='evict_last')
    gamma_1 = tl.load(input_ptr + (1 + 2 * x0), x_mask, eviction_policy='evict_last')
    mean = tl.load(mean_ptr + (x2), x_mask)
    gamma_2 = tl.load(input_ptr + (2 * x2), x_mask, eviction_policy='evict_last')
    gamma_3 = tl.load(input_ptr + (1 + 2 * x2), x_mask, eviction_policy='evict_last')
    variance = tl.load(variance_ptr + (x2), x_mask)
    
    gamma_input_0 = grad_input_0 * gamma_0
    gamma_input_1 = grad_input_1 * gamma_1
    sum_gamma_input = gamma_input_0 + gamma_input_1
    normalized_grad = sum_gamma_input * mean
    
    gamma_input_2 = gamma_2 * gamma_0
    gamma_input_3 = gamma_3 * gamma_1
    sum_gamma_2_3 = gamma_input_2 + gamma_input_3
    diff_grad = normalized_grad - sum_gamma_2_3
    
    variance_scaled = diff_grad * variance
    variance_scaled_cubed = variance_scaled * variance_scaled * variance_scaled
    
    neg_two = -2.0
    kernel_size_float = kernel_size.to(tl.float32)
    neg_two_plus_k = neg_two + kernel_size_float
    two = 2.0
    power_result = tl.extra.cuda.libdevice.pow(neg_two_plus_k, two)
    scaled_power = two * power_result
    scaled_power_double = scaled_power.to(tl.float64)
    
    one = tl.full([1], 1.0, tl.float64)
    inv_scaled_power = one / scaled_power_double
    inv_scaled_power_float = inv_scaled_power.to(tl.float32)
    
    scaled_variance = variance_scaled_cubed * inv_scaled_power_float
    neg_scaled_variance = -scaled_variance
    scaled_mean = neg_scaled_variance * mean
    
    variance_scaled_mean = variance_scaled * inv_scaled_power_float
    result = scaled_mean - variance_scaled_mean
    
    tl.store(output_grad_ptr + (x2), scaled_variance, x_mask)
    tl.store(output_ptr + (x2), result, x_mask)