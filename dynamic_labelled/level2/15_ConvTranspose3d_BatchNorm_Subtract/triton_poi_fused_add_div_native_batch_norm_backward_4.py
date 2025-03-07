# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_native_batch_norm_backward_4(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, 
    kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    batch_index = index // kernel_size0
    channel_index = ((index // kernel_size1) % 32)
    
    input_data = tl.load(in_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    mean_data = tl.load(in_ptr1 + (batch_index), mask, eviction_policy='evict_last')
    grad_output = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    variance_data = tl.load(in_ptr2 + (channel_index), mask, eviction_policy='evict_last')
    inv_std_data = tl.load(in_ptr3 + (channel_index), mask, eviction_policy='evict_last')
    grad_variance = tl.load(in_ptr4 + (channel_index), mask, eviction_policy='evict_last')
    grad_mean = tl.load(in_ptr5 + (channel_index), mask, eviction_policy='evict_last')
    grad_input = tl.load(in_ptr6 + (channel_index), mask, eviction_policy='evict_last')
    
    kernel_size = kernel_size0
    kernel_size_float = kernel_size.to(tl.float32)
    normalized_mean = mean_data / kernel_size_float
    normalized_input = input_data + normalized_mean
    delta = grad_output - variance_data
    normalization_factor = tl.full([], 1.0, tl.float64) / ((15872 + ((-63488) * kernel_size2) + 63488 * kernel_size2 * kernel_size2) / 32)
    normalization_factor_float = normalization_factor.to(tl.float32)
    inv_std_scaled = inv_std_data * normalization_factor_float
    variance_scaled = grad_variance * grad_variance
    scaled_variance = inv_std_scaled * variance_scaled
    delta_scaled = delta * scaled_variance
    adjusted_input = normalized_input - delta_scaled
    adjusted_mean = adjusted_input - inv_std_data * grad_mean
    grad_input_scaled = grad_input * inv_std_data
    final_gradient = adjusted_mean * grad_input_scaled
    
    tl.store(in_out_ptr0 + (linear_index), final_gradient, mask)