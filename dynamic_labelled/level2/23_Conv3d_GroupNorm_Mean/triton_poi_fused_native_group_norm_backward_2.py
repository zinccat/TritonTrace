# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_2(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    element_offset = index % 8

    input0_2x = tl.load(in_ptr0 + (2 * element_index), mask, eviction_policy='evict_last')
    input1_2x = tl.load(in_ptr1 + (2 * element_offset), mask, eviction_policy='evict_last')
    input0_2x_plus_1 = tl.load(in_ptr0 + (1 + 2 * element_index), mask, eviction_policy='evict_last')
    input1_2x_plus_1 = tl.load(in_ptr1 + (1 + 2 * element_offset), mask, eviction_policy='evict_last')
    in_out_value = tl.load(in_out_ptr0 + (element_index), mask)
    input2_2x = tl.load(in_ptr2 + (2 * element_index), mask, eviction_policy='evict_last')
    input2_2x_plus_1 = tl.load(in_ptr2 + (1 + 2 * element_index), mask, eviction_policy='evict_last')
    input3_value = tl.load(in_ptr3 + (element_index), mask)

    product0 = input0_2x * input1_2x
    product1 = input0_2x_plus_1 * input1_2x_plus_1
    sum_products = product0 + product1
    scaled_sum_products = sum_products * in_out_value
    product2 = input2_2x * input1_2x
    product3 = input2_2x_plus_1 * input1_2x_plus_1
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
    scaled_sum_products_input3 = sum_products * input3_value
    scaled_input3 = scaled_sum_products_input3 * reciprocal_float
    final_difference = scaled_difference_in_out - scaled_input3

    tl.store(out_ptr0 + (element_index), scaled_cubed_difference, mask)
    tl.store(in_out_ptr0 + (element_index), final_difference, mask)