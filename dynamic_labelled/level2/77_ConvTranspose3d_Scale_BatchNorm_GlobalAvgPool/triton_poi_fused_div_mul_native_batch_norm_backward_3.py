# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_mul_native_batch_norm_backward_3(
    in_out_grad_ptr, input_ptr, mean_ptr, variance_ptr, weight_ptr, running_mean_ptr, running_var_ptr,
    kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements

    batch_index = index // kernel_size_0
    element_index = index
    channel_index = ((index // kernel_size_1) % 32)

    input_grad = tl.load(input_ptr + (batch_index), mask, eviction_policy='evict_last')
    grad_output = tl.load(in_out_grad_ptr + (element_index), mask, eviction_policy='evict_last')
    mean = tl.load(mean_ptr + (channel_index), mask, eviction_policy='evict_last')
    variance = tl.load(variance_ptr + (channel_index), mask, eviction_policy='evict_last')
    weight = tl.load(weight_ptr + (channel_index), mask, eviction_policy='evict_last')
    running_mean = tl.load(running_mean_ptr + (channel_index), mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (channel_index), mask, eviction_policy='evict_last')

    scale_factor = kernel_size_0
    scale_factor_float = scale_factor.to(tl.float32)

    normalized_input = input_grad / scale_factor_float
    grad_output_scaled = grad_output * 2.0
    delta = grad_output_scaled - mean

    normalization_factor = (
        tl.full([], 1.0, tl.float64) /
        ((128 * kernel_size_2 * kernel_size_2 + 256 * kernel_size_2 + 32 * kernel_size_2 * kernel_size_2 * kernel_size_3 * kernel_size_3 +
          64 * kernel_size_2 * kernel_size_3 * kernel_size_3 + 128 * kernel_size_3 * kernel_size_2 * kernel_size_2 +
          256 * kernel_size_2 * kernel_size_3) / 32)
    )
    normalization_factor_float = normalization_factor.to(tl.float32)

    variance_scaled = variance * normalization_factor_float
    variance_squared = variance * variance
    variance_scaled_squared = variance_scaled * variance_squared
    delta_scaled = delta * variance_scaled_squared
    input_grad_adjusted = normalized_input - delta_scaled

    running_var_scaled = running_var * normalization_factor_float
    input_grad_final = input_grad_adjusted - running_var_scaled

    weight_scaled = weight * running_var
    grad_input = input_grad_final * weight_scaled * 2.0

    tl.store(in_out_grad_ptr + (element_index), grad_input, mask)