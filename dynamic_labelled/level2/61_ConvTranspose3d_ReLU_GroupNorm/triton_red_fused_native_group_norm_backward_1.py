# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_1red_fused_native_group_norm_backward_1(
    input_ptr, output_ptr, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices = input_index
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_indices = reduction_index
        
        temp_load = tl.load(
            input_ptr + (
                reduction_indices + 
                8 * input_indices + 
                2 * input_indices * kernel_size0 * kernel_size0 + 
                4 * kernel_size1 * input_indices + 
                8 * kernel_size0 * input_indices + 
                kernel_size1 * input_indices * kernel_size0 * kernel_size0 + 
                4 * kernel_size0 * kernel_size1 * input_indices
            ), 
            reduction_mask & input_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_sum, temp_accumulator)
    
    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_indices), temp_result, input_mask)