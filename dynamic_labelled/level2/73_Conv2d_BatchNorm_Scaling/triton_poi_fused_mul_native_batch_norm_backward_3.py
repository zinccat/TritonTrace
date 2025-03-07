# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_native_batch_norm_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, 
    kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size0) % 16)
    
    input_grad = tl.load(in_ptr0 + (x3), x_mask, eviction_policy='evict_last')
    output_grad = tl.load(in_out_ptr0 + (x3), x_mask, eviction_policy='evict_last')
    mean = tl.load(in_ptr1 + (x1), x_mask, eviction_policy='evict_last')
    inv_std = tl.load(in_ptr2 + (x1), x_mask, eviction_policy='evict_last')
    var = tl.load(in_ptr3 + (x1), x_mask, eviction_policy='evict_last')
    gamma = tl.load(in_ptr4 + (x1), x_mask, eviction_policy='evict_last')
    beta = tl.load(in_ptr5 + (x1), x_mask, eviction_policy='evict_last')
    
    scale_factor = 2.0
    scaled_input_grad = input_grad * scale_factor
    grad_diff = output_grad - mean
    
    normalization_factor = (
        tl.full([], 1.0, tl.float64) / 
        ((64 * kernel_size1 + ((-64) * kernel_size1 * kernel_size2) + 16 * kernel_size1 * kernel_size2 * kernel_size2) / 16)
    )
    normalization_factor = normalization_factor.to(tl.float32)
    
    scaled_inv_std = inv_std * normalization_factor
    var_squared = var * var
    scaled_var = scaled_inv_std * var_squared
    grad_scaled_var = grad_diff * scaled_var
    adjusted_grad = scaled_input_grad - grad_scaled_var
    
    scaled_gamma = gamma * normalization_factor
    adjusted_grad_gamma = adjusted_grad - scaled_gamma
    
    grad_beta = var * beta
    final_grad = adjusted_grad_gamma * grad_beta
    
    tl.store(in_out_ptr0 + (x3), final_grad, x_mask)