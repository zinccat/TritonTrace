# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_4(
    input_grad_ptr, input_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr, save_mean_ptr, 
    output_grad_ptr, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, num_elements, 
    XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements

    linear_index = index
    batch_index = index // kernel_size_0
    channel_index = ((index // kernel_size_1) % 128)
    kernel_index_0 = (index % kernel_size_1)
    kernel_index_1 = index // kernel_size_1

    input_grad = tl.load(input_grad_ptr + (linear_index), mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr + (batch_index), mask, eviction_policy='evict_last')
    weight_data = tl.load(weight_ptr + (channel_index), mask, eviction_policy='evict_last')
    bias_data = tl.load(bias_ptr + (
        2 * (((kernel_index_0 // (2 + kernel_size_2)) % (2 + kernel_size_2))) +
        4 * (kernel_index_0 // (4 + kernel_size_2 * kernel_size_2 + 4 * kernel_size_2)) +
        8 * kernel_index_1 +
        kernel_size_2 * (((kernel_index_0 // (2 + kernel_size_2)) % (2 + kernel_size_2))) +
        kernel_size_2 * kernel_size_2 * (kernel_index_0 // (4 + kernel_size_2 * kernel_size_2 + 4 * kernel_size_2)) +
        2 * kernel_index_1 * kernel_size_2 * kernel_size_2 +
        4 * kernel_size_2 * (kernel_index_0 // (4 + kernel_size_2 * kernel_size_2 + 4 * kernel_size_2)) +
        4 * kernel_size_3 * kernel_index_1 +
        8 * kernel_size_2 * kernel_index_1 +
        kernel_size_3 * kernel_index_1 * kernel_size_2 * kernel_size_2 +
        4 * kernel_size_2 * kernel_size_3 * kernel_index_1 +
        ((kernel_index_0 % (2 + kernel_size_2)))
    ), mask, eviction_policy='evict_last')
    running_mean = tl.load(running_mean_ptr + (batch_index), mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (batch_index), mask, eviction_policy='evict_last')
    save_mean = tl.load(save_mean_ptr + (batch_index), mask, eviction_policy='evict_last')

    weight_input_product = input_data * weight_data
    input_weight_product_grad = input_grad * weight_input_product

    zero = tl.full([1], 0, tl.int32)
    max_value = triton_helpers.maximum(zero, bias_data)

    running_mean_product = running_mean * running_var
    var_diff = running_mean_product - save_mean
    var_diff_weighted = var_diff * input_data
    var_diff_weighted_cubed = var_diff_weighted * var_diff_weighted * var_diff_weighted

    scale_factor_1 = 2.0
    kernel_size_2_float = kernel_size_2.to(tl.float32)
    scale_factor_2 = scale_factor_1 + kernel_size_2_float
    scale_factor_3 = tl.extra.cuda.libdevice.pow(scale_factor_2, scale_factor_1)
    scale_factor_4 = 16.0
    scale_factor_5 = scale_factor_4 * scale_factor_3
    kernel_size_3_float = kernel_size_3.to(tl.float32)
    scale_factor_6 = scale_factor_1 + kernel_size_3_float
    scale_factor_7 = scale_factor_5 * scale_factor_6
    scale_factor_8 = scale_factor_7.to(tl.float64)
    scale_factor_9 = tl.full([1], 1.0, tl.float64)
    scale_factor_10 = scale_factor_9 / scale_factor_8
    scale_factor_11 = scale_factor_10.to(tl.float32)

    var_diff_weighted_scaled = var_diff_weighted_cubed * scale_factor_11
    max_value_scaled = max_value * var_diff_weighted_scaled
    output_grad = input_weight_product_grad + max_value_scaled

    tl.store(output_grad_ptr + (linear_index), output_grad, mask)