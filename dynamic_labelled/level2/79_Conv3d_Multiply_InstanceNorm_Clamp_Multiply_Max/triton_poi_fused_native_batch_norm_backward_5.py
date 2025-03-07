# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_batch_norm_backward_5poi_fused_native_batch_norm_backward_5(
    in_out_ptr, input_grad, input_data, scale, bias, running_var, running_mean, 
    kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4, 
    input_num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < input_num_elements
    x2 = x_index
    x1 = x_index // kernel_size_1

    grad_output = tl.load(in_out_ptr + (x2), x_mask, eviction_policy='evict_last')
    input_data_value = tl.load(input_data + (x2), x_mask, eviction_policy='evict_last')
    scale_value = tl.load(scale + (((x2 // kernel_size_0) % 16)), x_mask, eviction_policy='evict_last')
    bias_value = tl.load(bias + (x1), x_mask, eviction_policy='evict_last')
    running_var_value = tl.load(running_var + (x1), x_mask, eviction_policy='evict_last')
    running_mean_value = tl.load(running_mean + (x1), x_mask, eviction_policy='evict_last')

    scaled_input = input_data_value * scale_value
    normalized_input = scaled_input - bias_value

    normalization_factor = (
        tl.full([], 1.00000000000000, tl.float64) / 
        ((((-128) * kernel_size_2) + ((-32) * kernel_size_2 * kernel_size_4 * kernel_size_4) + 
         64 * kernel_size_2 * kernel_size_3 + 128 * kernel_size_2 * kernel_size_4 + 
         ((-64) * kernel_size_2 * kernel_size_3 * kernel_size_4) + 
         16 * kernel_size_2 * kernel_size_3 * kernel_size_4 * kernel_size_4) / 
         (16 * kernel_size_2))
    )
    normalization_factor = normalization_factor.to(tl.float32)

    mean_adjusted = running_mean_value * normalization_factor
    variance_adjusted = running_var_value * running_var_value
    std_dev_adjusted = mean_adjusted * variance_adjusted

    grad_input = normalized_input * std_dev_adjusted
    grad_input_adjusted = grad_output - grad_input

    mean_grad = running_mean_value * normalization_factor
    grad_input_final = grad_input_adjusted - mean_grad

    grad_scale = grad_input_final * running_var_value
    tl.store(in_out_ptr + (x2), grad_scale, x_mask)