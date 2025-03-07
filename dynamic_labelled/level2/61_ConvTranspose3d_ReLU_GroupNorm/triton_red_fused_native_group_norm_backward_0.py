# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_0red_fused_native_group_norm_backward_0(
    input_grad_ptr, input_ptr, output_ptr, kernel_size_0, kernel_size_1, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < reduction_num_elements
        r1 = r_index
        
        input_grad_index = (
            2 * (((r1 // (2 + kernel_size_0)) % (2 + kernel_size_0))) +
            4 * (r1 // (4 + kernel_size_0 * kernel_size_0 + 4 * kernel_size_0)) +
            8 * x0 +
            kernel_size_0 * (((r1 // (2 + kernel_size_0)) % (2 + kernel_size_0))) +
            kernel_size_0 * kernel_size_0 * (r1 // (4 + kernel_size_0 * kernel_size_0 + 4 * kernel_size_0)) +
            2 * x0 * kernel_size_0 * kernel_size_0 +
            4 * kernel_size_0 * (r1 // (4 + kernel_size_0 * kernel_size_0 + 4 * kernel_size_0)) +
            4 * kernel_size_1 * x0 +
            8 * kernel_size_0 * x0 +
            kernel_size_1 * x0 * kernel_size_0 * kernel_size_0 +
            4 * kernel_size_0 * kernel_size_1 * x0 +
            ((r1 % (2 + kernel_size_0)))
        )
        
        grad_input = tl.load(input_grad_ptr + input_grad_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_value = tl.load(input_ptr + input_grad_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        
        zero_tensor = tl.full([1, 1], 0, tl.int32)
        max_value = triton_helpers.maximum(zero_tensor, input_value)
        element_wise_product = grad_input * max_value
        broadcasted_product = tl.broadcast_to(element_wise_product, [XBLOCK, RBLOCK])
        temp_sum_update = temp_sum + broadcasted_product
        
        temp_sum = tl.where(r_mask & x_mask, temp_sum_update, temp_sum)
    
    reduced_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr + (x0), reduced_sum, x_mask)