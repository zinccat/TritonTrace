# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_4(
    input_grad_ptr, input_ptr, scale_ptr, output_grad_ptr0, output_grad_ptr1, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x0 = (x_index % 16)
    
    input_scale = tl.load(scale_ptr + (x0), x_mask, eviction_policy='evict_last')
    reduction_sum1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    reduction_sum2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < reduction_num_elements
        r2 = r_index
        
        input_grad = tl.load(
            input_grad_ptr + (((-2) * (r2 // kernel_size0)) + 4 * x3 + kernel_size1 * (r2 // kernel_size0) + x3 * kernel_size1 * kernel_size1 + ((-4) * kernel_size1 * x3) + ((r2 % kernel_size0))),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        
        scale_value = tl.load(
            scale_ptr + (((-2) * (r2 // kernel_size0)) + 4 * x3 + kernel_size1 * (r2 // kernel_size0) + x3 * kernel_size1 * kernel_size1 + ((-4) * kernel_size1 * x3) + ((r2 % kernel_size0))),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        
        product1 = input_grad * input_scale
        product2 = product1 * scale_value
        broadcast_product2 = tl.broadcast_to(product2, [XBLOCK, RBLOCK])
        reduction_sum1_update = reduction_sum1 + broadcast_product2
        reduction_sum1 = tl.where(r_mask & x_mask, reduction_sum1_update, reduction_sum1)
        
        broadcast_product1 = tl.broadcast_to(product1, [XBLOCK, RBLOCK])
        reduction_sum2_update = reduction_sum2 + broadcast_product1
        reduction_sum2 = tl.where(r_mask & x_mask, reduction_sum2_update, reduction_sum2)
    
    reduction_sum1_final = tl.sum(reduction_sum1, 1)[:, None]
    reduction_sum2_final = tl.sum(reduction_sum2, 1)[:, None]
    
    tl.store(output_grad_ptr0 + (x3), reduction_sum1_final, x_mask)
    tl.store(output_grad_ptr1 + (x3), reduction_sum2_final, x_mask)