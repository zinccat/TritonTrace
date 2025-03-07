# From: 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_mean_0(input_ptr0, input_ptr1, output_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_3d_index = input_index
    input_0d_index = (input_index % 16)
    
    input_data_1 = tl.load(input_ptr1 + (input_0d_index), input_mask, eviction_policy='evict_last')
    accumulated_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_2d_index = reduction_index
        
        input_data_0 = tl.load(input_ptr0 + (reduction_2d_index + 4 * input_3d_index + input_3d_index * kernel_size * kernel_size + 4 * kernel_size * input_3d_index), reduction_mask & input_mask, eviction_policy='evict_first', other=0.0)
        combined_data = input_data_0 + input_data_1
        broadcasted_data = tl.broadcast_to(combined_data, [XBLOCK, RBLOCK])
        updated_sum = accumulated_sum + broadcasted_data
        accumulated_sum = tl.where(reduction_mask & input_mask, updated_sum, accumulated_sum)
    
    summed_result = tl.sum(accumulated_sum, 1)[:, None]
    tl.store(output_ptr0 + (input_3d_index), summed_result, input_mask)