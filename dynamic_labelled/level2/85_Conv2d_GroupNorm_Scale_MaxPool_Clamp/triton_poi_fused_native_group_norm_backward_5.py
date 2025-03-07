# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_5(input_grad_ptr, input_ptr, mean_ptr, variance_ptr, scale_ptr, output_grad_ptr, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x2 = x_index
    x0 = (x_index % 8)
    
    input_grad_0 = tl.load(input_grad_ptr + (2 * x2), x_mask, eviction_policy='evict_last')
    input_0 = tl.load(input_ptr + (2 * x0), x_mask, eviction_policy='evict_last')
    input_grad_1 = tl.load(input_grad_ptr + (1 + 2 * x2), x_mask, eviction_policy='evict_last')
    input_1 = tl.load(input_ptr + (1 + 2 * x0), x_mask, eviction_policy='evict_last')
    mean = tl.load(mean_ptr + (x2), x_mask)
    variance_0 = tl.load(variance_ptr + (2 * x2), x_mask, eviction_policy='evict_last')
    variance_1 = tl.load(variance_ptr + (1 + 2 * x2), x_mask, eviction_policy='evict_last')
    scale = tl.load(scale_ptr + (x2), x_mask)
    
    grad_input_0 = input_grad_0 * input_0
    grad_input_1 = input_grad_1 * input_1
    sum_grad_input = grad_input_0 + grad_input_1
    normalized_grad = sum_grad_input * mean
    
    variance_grad_0 = variance_0 * input_0
    variance_grad_1 = variance_1 * input_1
    sum_variance_grad = variance_grad_0 + variance_grad_1
    variance_correction = normalized_grad - sum_variance_grad
    
    scaled_variance_correction = variance_correction * scale
    cubed_scaled_variance_correction = scaled_variance_correction * scaled_variance_correction * scaled_variance_correction
    
    neg_two = -2.0
    kernel_size_float = kernel_size.to(tl.float32)
    power_base = neg_two + kernel_size_float
    power_result = tl.extra.cuda.libdevice.pow(power_base, 2.0)
    power_scaled = 2.0 * power_result
    power_scaled_double = power_scaled.to(tl.float64)
    
    one = tl.full([1], 1.0, tl.float64)
    inv_power_scaled = one / power_scaled_double
    inv_power_scaled_float = inv_power_scaled.to(tl.float32)
    
    adjusted_grad = cubed_scaled_variance_correction * inv_power_scaled_float
    neg_adjusted_grad = -adjusted_grad
    scaled_adjusted_grad = neg_adjusted_grad * mean
    
    final_variance_correction = scaled_variance_correction * inv_power_scaled_float
    final_output = scaled_adjusted_grad - final_variance_correction
    
    tl.store(output_grad_ptr + (x2), adjusted_grad, x_mask)
    tl.store(output_ptr + (x2), final_output, x_mask)