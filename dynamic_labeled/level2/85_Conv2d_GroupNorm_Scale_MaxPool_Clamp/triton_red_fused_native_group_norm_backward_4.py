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
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_3d = x_indices
    x_indices_0d = (x_indices % 16)
    
    scale_values = tl.load(scale_ptr + (x_indices_0d), x_mask, eviction_policy='evict_last')
    sum_grad_output = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_input_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_2d = r_indices
        
        input_grad_values = tl.load(
            input_grad_ptr + (((-2) * (r_indices_2d // kernel_size0)) + 4 * x_indices_3d + kernel_size1 * (r_indices_2d // kernel_size0) + x_indices_3d * kernel_size1 * kernel_size1 + ((-4) * kernel_size1 * x_indices_3d) + ((r_indices_2d % kernel_size0))),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        
        scale_values_at_indices = tl.load(
            scale_ptr + (((-2) * (r_indices_2d // kernel_size0)) + 4 * x_indices_3d + kernel_size1 * (r_indices_2d // kernel_size0) + x_indices_3d * kernel_size1 * kernel_size1 + ((-4) * kernel_size1 * x_indices_3d) + ((r_indices_2d % kernel_size0))),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        
        input_grad_scaled = input_grad_values * scale_values
        grad_output_scaled = input_grad_scaled * scale_values_at_indices
        
        grad_output_broadcast = tl.broadcast_to(grad_output_scaled, [XBLOCK, RBLOCK])
        sum_grad_output += grad_output_broadcast
        sum_grad_output = tl.where(r_mask & x_mask, sum_grad_output, sum_grad_output)
        
        input_grad_broadcast = tl.broadcast_to(input_grad_scaled, [XBLOCK, RBLOCK])
        sum_input_grad += input_grad_broadcast
        sum_input_grad = tl.where(r_mask & x_mask, sum_input_grad, sum_input_grad)
    
    sum_grad_output_reduced = tl.sum(sum_grad_output, 1)[:, None]
    sum_input_grad_reduced = tl.sum(sum_input_grad, 1)[:, None]
    
    tl.store(output_grad_ptr0 + (x_indices_3d), sum_grad_output_reduced, x_mask)
    tl.store(output_grad_ptr1 + (x_indices_3d), sum_input_grad_reduced, x_mask)