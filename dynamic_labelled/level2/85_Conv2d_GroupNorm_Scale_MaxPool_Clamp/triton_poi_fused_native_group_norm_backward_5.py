# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_5(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, 
    output_grad_ptr, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
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
    
    grad_gamma_0 = input_grad_0 * gamma_0
    grad_gamma_1 = input_grad_1 * gamma_1
    grad_input = grad_gamma_0 + grad_gamma_1
    grad_input_scaled = grad_input * mean
    
    gamma_grad_scaled_0 = gamma_grad_0 * gamma_0
    gamma_grad_scaled_1 = gamma_grad_1 * gamma_1
    gamma_grad_scaled = gamma_grad_scaled_0 + gamma_grad_scaled_1
    grad_input_scaled_diff = grad_input_scaled - gamma_grad_scaled
    
    grad_variance = grad_input_scaled_diff * variance
    grad_variance_cubed = grad_variance * grad_variance * grad_variance
    
    neg_two = -2.0
    kernel_size_float = kernel_size.to(tl.float32)
    neg_two_plus_kernel_size = neg_two + kernel_size_float
    two = 2.0
    power_result = tl.extra.cuda.libdevice.pow(neg_two_plus_kernel_size, two)
    two_times_power = two * power_result
    double_two_times_power = two_times_power.to(tl.float64)
    
    one = tl.full([1], 1.0, tl.float64)
    inv_double_two_times_power = one / double_two_times_power
    inv_double_two_times_power_float = inv_double_two_times_power.to(tl.float32)
    
    grad_variance_scaled = grad_variance_cubed * inv_double_two_times_power_float
    neg_grad_variance_scaled = -grad_variance_scaled
    
    grad_mean = neg_grad_variance_scaled * mean
    grad_input_scaled_neg = grad_mean - (grad_variance_scaled * inv_double_two_times_power_float)
    
    tl.store(output_grad_ptr + (x2), grad_variance_scaled, x_mask)
    tl.store(output_ptr + (x2), grad_input_scaled_neg, x_mask)