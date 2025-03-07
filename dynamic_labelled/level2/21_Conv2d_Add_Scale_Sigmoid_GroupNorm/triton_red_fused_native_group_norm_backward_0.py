# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_0(
    input_grad_ptr, input_ptr, mean_ptr, inv_std_ptr, output_grad_ptr, kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_3d = x_indices
    x_indices_0d = (x_indices % 16)
    
    mean_values = tl.load(mean_ptr + (x_indices_0d), x_mask, eviction_policy='evict_last')
    inv_std_values = tl.load(inv_std_ptr + (x_indices_0d), x_mask, eviction_policy='evict_last')
    
    accumulated_gradients = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_2d = r_indices
        
        input_grad_index = (
            (-2) * (triton_helpers.div_floor_integer(r_indices_2d, (-2) + kernel_size)) 
            + 4 * x_indices_3d 
            + kernel_size * (triton_helpers.div_floor_integer(r_indices_2d, (-2) + kernel_size)) 
            + x_indices_3d * kernel_size * kernel_size 
            + (-4) * kernel_size * x_indices_3d 
            + (r_indices_2d % ((-2) + kernel_size))
        )
        
        input_grad_values = tl.load(input_grad_ptr + input_grad_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_values = tl.load(input_ptr + input_grad_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        
        combined_values = input_values + mean_values
        scaled_values = combined_values * inv_std_values
        sigmoid_values = tl.sigmoid(scaled_values)
        
        element_wise_product = input_grad_values * sigmoid_values
        broadcasted_product = tl.broadcast_to(element_wise_product, [XBLOCK, RBLOCK])
        
        accumulated_gradients += broadcasted_product
        accumulated_gradients = tl.where(r_mask & x_mask, accumulated_gradients, accumulated_gradients)
    
    summed_gradients = tl.sum(accumulated_gradients, 1)[:, None]
    tl.store(output_grad_ptr + (x_indices_3d), summed_gradients, x_mask)