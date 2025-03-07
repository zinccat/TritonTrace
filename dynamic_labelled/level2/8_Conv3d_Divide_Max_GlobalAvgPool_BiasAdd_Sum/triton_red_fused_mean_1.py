# From: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_1red_fused_mean_1(input_ptr, output_ptr, kernel_size_0, kernel_size_1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_0 = input_index
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_1 = reduction_index
        
        temp_load = tl.load(
            input_ptr + (
                reduction_1 + 
                ((-1) * input_0) + 
                input_0 * (kernel_size_0 // 2) + 
                ((-1) * input_0 * (kernel_size_1 // 2) * (kernel_size_1 // 2)) + 
                2 * input_0 * (kernel_size_1 // 2) + 
                input_0 * (kernel_size_1 // 2) * (kernel_size_1 // 2) * (kernel_size_0 // 2) + 
                ((-2) * input_0 * (kernel_size_0 // 2) * (kernel_size_1 // 2))
            ), 
            reduction_mask & input_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_sum, temp_accumulator)
    
    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_0), temp_result, input_mask)