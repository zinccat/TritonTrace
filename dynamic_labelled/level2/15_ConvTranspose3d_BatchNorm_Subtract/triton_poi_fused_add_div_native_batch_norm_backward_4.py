# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_native_batch_norm_backward_4poi_fused_add_div_native_batch_norm_backward_4(
    in_out_ptr, input_grad, running_mean, running_var, weight, bias, inv_std, scale_factor, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    batch_index = index // kernel_size_0
    channel_index = ((index // kernel_size_1) % 32)
    
    input_grad_value = tl.load(input_grad + (linear_index), mask, eviction_policy='evict_last')
    running_mean_value = tl.load(running_mean + (batch_index), mask, eviction_policy='evict_last')
    in_out_value = tl.load(in_out_ptr + (linear_index), mask, eviction_policy='evict_last')
    weight_value = tl.load(weight + (channel_index), mask, eviction_policy='evict_last')
    bias_value = tl.load(bias + (channel_index), mask, eviction_policy='evict_last')
    inv_std_value = tl.load(inv_std + (channel_index), mask, eviction_policy='evict_last')
    scale_factor_value = tl.load(scale_factor + (channel_index), mask, eviction_policy='evict_last')
    
    kernel_size_0_float = kernel_size_0.to(tl.float32)
    normalized_mean = running_mean_value / kernel_size_0_float
    adjusted_input_grad = input_grad_value + normalized_mean
    
    mean_diff = in_out_value - bias_value
    
    normalization_factor = tl.full([], 1.0, tl.float64) / ((15872 + ((-63488) * kernel_size_2) + 63488 * kernel_size_2 * kernel_size_2) / 32)
    normalization_factor_float = normalization_factor.to(tl.float32)
    weight_scaled = weight_value * normalization_factor_float
    
    weight_squared = weight_value * weight_value
    weight_scaled_squared = weight_scaled * weight_squared
    
    adjusted_mean_diff = mean_diff * weight_scaled_squared
    adjusted_input_grad_diff = adjusted_input_grad - adjusted_mean_diff
    
    inv_std_scaled = inv_std_value * normalization_factor_float
    final_diff = adjusted_input_grad_diff - inv_std_scaled
    
    output_value = final_diff * (weight_value * scale_factor_value)
    
    tl.store(in_out_ptr + (linear_index), output_value, mask)