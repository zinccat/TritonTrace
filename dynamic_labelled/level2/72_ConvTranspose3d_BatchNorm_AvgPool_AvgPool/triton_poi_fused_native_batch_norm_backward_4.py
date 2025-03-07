# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_batch_norm_backward_4poi_fused_native_batch_norm_backward_4(
    in_out_grad_ptr, input_data_ptr, mean_ptr, inv_std_ptr, grad_output_ptr, scale_ptr, running_var_ptr,
    kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    batch_index = (index // kernel_size_0) % 16

    grad_output = tl.load(in_out_grad_ptr + (linear_index), mask, eviction_policy='evict_last')
    input_data = tl.load(input_data_ptr + (linear_index), mask, eviction_policy='evict_last')
    mean = tl.load(mean_ptr + (batch_index), mask, eviction_policy='evict_last')
    inv_std = tl.load(inv_std_ptr + (batch_index), mask, eviction_policy='evict_last')
    grad_input = tl.load(grad_output_ptr + (batch_index), mask, eviction_policy='evict_last')
    scale = tl.load(scale_ptr + (batch_index), mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (batch_index), mask, eviction_policy='evict_last')

    input_centered = input_data - mean
    normalization_factor = (
        tl.full([], 1.0, tl.float64) /
        (((-16 * kernel_size_1) + ((-192 * kernel_size_1 * kernel_size_2 * kernel_size_2) + 96 * kernel_size_1 * kernel_size_2 + 128 * kernel_size_1 * kernel_size_2 * kernel_size_2 * kernel_size_2) / 16))
    )
    normalization_factor = normalization_factor.to(tl.float32)
    scaled_inv_std = inv_std * normalization_factor
    variance_term = running_var * running_var
    scaled_variance = scaled_inv_std * variance_term
    grad_input_scaled = input_centered * scaled_variance
    grad_output_adjusted = grad_output - grad_input_scaled
    scaled_scale = scale * normalization_factor
    grad_output_final = grad_output_adjusted - scaled_scale
    running_var_scaled = running_var * scale
    final_gradient = grad_output_final * running_var_scaled

    tl.store(in_out_grad_ptr + (linear_index), final_gradient, mask)