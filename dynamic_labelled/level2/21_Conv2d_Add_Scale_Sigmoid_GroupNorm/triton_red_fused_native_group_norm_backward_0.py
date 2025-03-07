# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_0(
    input_grad_ptr, input_ptr, scale_ptr, running_var_ptr, output_grad_ptr, kernel_size, x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x0 = (x_index % 16)
    
    scale_value = tl.load(scale_ptr + (x0), x_mask, eviction_policy='evict_last')
    running_var_value = tl.load(running_var_ptr + (x0), x_mask, eviction_policy='evict_last')
    
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index
        
        input_grad_index = (
            (-2) * (triton_helpers.div_floor_integer(r2, (-2) + kernel_size))
            + 4 * x3
            + kernel_size * (triton_helpers.div_floor_integer(r2, (-2) + kernel_size))
            + x3 * kernel_size * kernel_size
            + (-4) * kernel_size * x3
            + (r2 % ((-2) + kernel_size))
        )
        
        input_grad_value = tl.load(input_grad_ptr + input_grad_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_value = tl.load(input_ptr + input_grad_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        
        combined_value = input_value + scale_value
        scaled_value = combined_value * running_var_value
        sigmoid_value = tl.sigmoid(scaled_value)
        element_wise_product = input_grad_value * sigmoid_value
        
        broadcasted_product = tl.broadcast_to(element_wise_product, [XBLOCK, RBLOCK])
        temp_sum = temp_sum + broadcasted_product
        
        temp_sum = tl.where(r_mask & x_mask, temp_sum, temp_sum)
    
    reduced_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_grad_ptr + (x3), reduced_sum, x_mask)