# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_2(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, weight_ptr, output_grad_ptr, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
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
    weight = tl.load(weight_ptr + (x2), x_mask)
    
    product_0 = input_grad_0 * input_0
    product_1 = input_grad_1 * input_1
    sum_products = product_0 + product_1
    scaled_sum = sum_products * mean
    variance_product_0 = variance_0 * input_0
    variance_product_1 = variance_1 * input_1
    sum_variance_products = variance_product_0 + variance_product_1
    difference = scaled_sum - sum_variance_products
    weighted_difference = difference * weight
    cubed_weighted_difference = weighted_difference * weighted_difference * weighted_difference
    
    neg_two = -2.0
    kernel_size_float = kernel_size.to(tl.float32)
    neg_two_plus_kernel_size = neg_two + kernel_size_float
    two = 2.0
    power_result = tl.extra.cuda.libdevice.pow(neg_two_plus_kernel_size, two)
    scaled_power = two * power_result
    double_scaled_power = scaled_power.to(tl.float64)
    one = tl.full([1], 1.0, tl.float64)
    reciprocal = one / double_scaled_power
    reciprocal_float = reciprocal.to(tl.float32)
    scaled_cubed_difference = cubed_weighted_difference * reciprocal_float
    neg_scaled_cubed_difference = -scaled_cubed_difference
    scaled_difference = neg_scaled_cubed_difference * mean
    scaled_weighted_sum = sum_products * weight
    scaled_reciprocal = scaled_weighted_sum * reciprocal_float
    final_difference = scaled_difference - scaled_reciprocal
    
    tl.store(output_grad_ptr + (x2), scaled_cubed_difference, x_mask)
    tl.store(output_ptr + (x2), final_difference, x_mask)