# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_native_batch_norm_backward_3poi_fused_mul_native_batch_norm_backward_3(
    in_out_ptr, input_grad_ptr, mean_ptr, inv_std_ptr, weight_ptr, running_var_ptr, weight_grad_ptr,
    kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size_0) % 16)
    
    input_grad = tl.load(input_grad_ptr + (x3), x_mask, eviction_policy='evict_last')
    in_out = tl.load(in_out_ptr + (x3), x_mask, eviction_policy='evict_last')
    mean = tl.load(mean_ptr + (x1), x_mask, eviction_policy='evict_last')
    inv_std = tl.load(inv_std_ptr + (x1), x_mask, eviction_policy='evict_last')
    weight = tl.load(weight_ptr + (x1), x_mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x1), x_mask, eviction_policy='evict_last')
    weight_grad = tl.load(weight_grad_ptr + (x1), x_mask, eviction_policy='evict_last')
    
    scale_factor = 2.0
    scaled_input_grad = input_grad * scale_factor
    grad_diff = in_out - mean
    normalization_factor = (
        tl.full([], 1.0, tl.float64) / 
        ((64 * kernel_size_1 + ((-64) * kernel_size_1 * kernel_size_2) + 16 * kernel_size_1 * kernel_size_2 * kernel_size_2) / 16)
    )
    normalization_factor = normalization_factor.to(tl.float32)
    scaled_inv_std = inv_std * normalization_factor
    variance_term = weight * weight
    scaled_variance = scaled_inv_std * variance_term
    variance_grad = grad_diff * scaled_variance
    adjusted_grad = scaled_input_grad - variance_grad
    scaled_running_var = running_var * normalization_factor
    final_grad = adjusted_grad - scaled_running_var
    weight_grad_term = weight * weight_grad
    output_grad = final_grad * weight_grad_term
    
    tl.store(in_out_ptr + (x3), output_grad, x_mask)