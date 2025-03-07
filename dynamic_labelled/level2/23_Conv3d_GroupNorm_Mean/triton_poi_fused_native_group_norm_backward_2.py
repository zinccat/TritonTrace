# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_2poi_fused_native_group_norm_backward_2(
    in_out_ptr, input_grad_ptr, input_data_ptr, mean_ptr, variance_ptr, output_grad_ptr, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    element_offset = index % 8

    input_grad_0 = tl.load(input_grad_ptr + (2 * element_index), mask, eviction_policy='evict_last')
    input_data_0 = tl.load(input_data_ptr + (2 * element_offset), mask, eviction_policy='evict_last')
    input_grad_1 = tl.load(input_grad_ptr + (1 + 2 * element_index), mask, eviction_policy='evict_last')
    input_data_1 = tl.load(input_data_ptr + (1 + 2 * element_offset), mask, eviction_policy='evict_last')
    in_out_value = tl.load(in_out_ptr + (element_index), mask)
    mean_value = tl.load(mean_ptr + (element_index), mask)
    variance_0 = tl.load(variance_ptr + (2 * element_index), mask, eviction_policy='evict_last')
    variance_1 = tl.load(variance_ptr + (1 + 2 * element_index), mask, eviction_policy='evict_last')

    grad_input_data_0 = input_grad_0 * input_data_0
    grad_input_data_1 = input_grad_1 * input_data_1
    sum_grad_input_data = grad_input_data_0 + grad_input_data_1
    scaled_sum_grad_input_data = sum_grad_input_data * in_out_value
    variance_scaled_input_data_0 = variance_0 * input_data_0
    variance_scaled_input_data_1 = variance_1 * input_data_1
    sum_variance_scaled_input_data = variance_scaled_input_data_0 + variance_scaled_input_data_1
    variance_correction = scaled_sum_grad_input_data - sum_variance_scaled_input_data
    variance_correction_scaled = variance_correction * mean_value
    variance_correction_cubed = variance_correction_scaled * variance_correction_scaled * variance_correction_scaled

    neg_two = -2.0
    kernel_size0_float = kernel_size0.to(tl.float32)
    kernel_size0_adjusted = neg_two + kernel_size0_float
    power_two = 2.0
    kernel_size0_power = tl.extra.cuda.libdevice.pow(kernel_size0_adjusted, power_two)
    kernel_size0_scaled = power_two * kernel_size0_power
    kernel_size1_float = kernel_size1.to(tl.float32)
    kernel_size1_adjusted = kernel_size0_scaled * (neg_two + kernel_size1_float)
    kernel_size1_scaled = kernel_size1_adjusted.to(tl.float64)

    one = tl.full([1], 1.0, tl.float64)
    normalization_factor = one / kernel_size1_scaled
    normalization_factor_float = normalization_factor.to(tl.float32)

    corrected_variance = variance_correction_cubed * normalization_factor_float
    neg_corrected_variance = -corrected_variance
    scaled_in_out_value = neg_corrected_variance * in_out_value
    mean_scaled_sum_grad_input_data = sum_grad_input_data * mean_value
    mean_scaled_correction = mean_scaled_sum_grad_input_data * normalization_factor_float
    final_in_out_value = scaled_in_out_value - mean_scaled_correction

    tl.store(output_grad_ptr + (element_index), corrected_variance, mask)
    tl.store(in_out_ptr + (element_index), final_in_out_value, mask)