# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_2poi_fused_native_group_norm_backward_2(
    input_grad_ptr, input_ptr, mean_ptr, var_ptr, weight_ptr, output_grad_ptr, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x2 = x_index
    x0 = (x_index % 8)
    
    grad_input_0 = tl.load(input_grad_ptr + (2 * x2), x_mask, eviction_policy='evict_last')
    weight_0 = tl.load(input_ptr + (2 * x0), x_mask, eviction_policy='evict_last')
    grad_input_1 = tl.load(input_grad_ptr + (1 + 2 * x2), x_mask, eviction_policy='evict_last')
    weight_1 = tl.load(input_ptr + (1 + 2 * x0), x_mask, eviction_policy='evict_last')
    mean = tl.load(mean_ptr + (x2), x_mask)
    weight_2 = tl.load(var_ptr + (2 * x2), x_mask, eviction_policy='evict_last')
    weight_3 = tl.load(var_ptr + (1 + 2 * x2), x_mask, eviction_policy='evict_last')
    var = tl.load(weight_ptr + (x2), x_mask)
    
    grad_input_weight_0 = grad_input_0 * weight_0
    grad_input_weight_1 = grad_input_1 * weight_1
    grad_input_sum = grad_input_weight_0 + grad_input_weight_1
    
    mean_scaled = grad_input_sum * mean
    weight_2_weight_0 = weight_2 * weight_0
    weight_3_weight_1 = weight_3 * weight_1
    weight_sum = weight_2_weight_0 + weight_3_weight_1
    
    grad_input_mean_diff = mean_scaled - weight_sum
    var_scaled = grad_input_mean_diff * var
    var_cubed = var_scaled * var_scaled * var_scaled
    
    neg_two = -2.0
    kernel_size_float = kernel_size.to(tl.float32)
    neg_two_plus_kernel_size = neg_two + kernel_size_float
    two = 2.0
    power_result = tl.extra.cuda.libdevice.pow(neg_two_plus_kernel_size, two)
    two_times_power = two * power_result
    double_two_times_power = two_times_power.to(tl.float64)
    
    one = tl.full([1], 1.0, tl.float64)
    reciprocal = one / double_two_times_power
    reciprocal_float = reciprocal.to(tl.float32)
    
    var_cubed_scaled = var_cubed * reciprocal_float
    neg_var_cubed_scaled = -var_cubed_scaled
    neg_var_cubed_scaled_mean = neg_var_cubed_scaled * mean
    var_scaled_reciprocal = var_scaled * reciprocal_float
    grad_input_var_diff = neg_var_cubed_scaled_mean - var_scaled_reciprocal
    
    tl.store(output_grad_ptr + (x2), var_cubed_scaled, x_mask)
    tl.store(output_ptr + (x2), grad_input_var_diff, x_mask)