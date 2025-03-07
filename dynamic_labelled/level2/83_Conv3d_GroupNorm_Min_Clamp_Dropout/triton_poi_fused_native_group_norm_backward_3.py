# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, kernel_size0, kernel_size1, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 8)
    
    input0_2x = tl.load(in_ptr0 + (2 * x2), xmask, eviction_policy='evict_last')
    input1_2x = tl.load(in_ptr1 + (2 * x0), xmask, eviction_policy='evict_last')
    input0_2x_plus1 = tl.load(in_ptr0 + (1 + 2 * x2), xmask, eviction_policy='evict_last')
    input1_2x_plus1 = tl.load(in_ptr1 + (1 + 2 * x0), xmask, eviction_policy='evict_last')
    in_out_value = tl.load(in_out_ptr0 + (x2), xmask)
    input2_2x = tl.load(in_ptr2 + (2 * x2), xmask, eviction_policy='evict_last')
    input2_2x_plus1 = tl.load(in_ptr2 + (1 + 2 * x2), xmask, eviction_policy='evict_last')
    input3_value = tl.load(in_ptr3 + (x2), xmask)
    
    product0 = input0_2x * input1_2x
    product1 = input0_2x_plus1 * input1_2x_plus1
    sum_products = product0 + product1
    scaled_sum_products = sum_products * in_out_value
    product2 = input2_2x * input1_2x
    product3 = input2_2x_plus1 * input1_2x_plus1
    sum_products2 = product2 + product3
    difference = scaled_sum_products - sum_products2
    scaled_difference = difference * input3_value
    cubed_difference = scaled_difference * scaled_difference * scaled_difference
    
    neg_two = -2.0
    kernel_size0_float = kernel_size0.to(tl.float32)
    adjusted_kernel_size0 = neg_two + kernel_size0_float
    power_two = 2.0
    power_result = tl.extra.cuda.libdevice.pow(adjusted_kernel_size0, power_two)
    scaled_power_result = power_two * power_result
    kernel_size1_float = kernel_size1.to(tl.float32)
    adjusted_kernel_size1 = neg_two + kernel_size1_float
    final_scale = scaled_power_result * adjusted_kernel_size1
    final_scale_double = final_scale.to(tl.float64)
    one_double = tl.full([1], 1.0, tl.float64)
    reciprocal = one_double / final_scale_double
    reciprocal_float = reciprocal.to(tl.float32)
    scaled_cubed_difference = cubed_difference * reciprocal_float
    neg_scaled_cubed_difference = -scaled_cubed_difference
    scaled_difference_in_out = neg_scaled_cubed_difference * in_out_value
    scaled_sum_products_scaled = sum_products * input3_value
    scaled_scaled_sum_products = scaled_sum_products_scaled * reciprocal_float
    final_difference = scaled_difference_in_out - scaled_scaled_sum_products
    
    tl.store(out_ptr0 + (x2), scaled_cubed_difference, xmask)
    tl.store(in_out_ptr0 + (x2), final_difference, xmask)