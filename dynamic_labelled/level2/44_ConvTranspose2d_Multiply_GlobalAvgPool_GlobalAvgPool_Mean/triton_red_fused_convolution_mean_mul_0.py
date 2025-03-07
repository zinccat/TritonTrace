# From: 44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_mean_mul_0(in_out_ptr0, in_ptr0, in_ptr1, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_3d_indices = input_indices
    input_0d_indices = (input_indices % 16)
    
    input_1_values = tl.load(in_ptr1 + (input_0d_indices), input_mask, eviction_policy='evict_last')
    accumulated_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_2d_indices = reduction_indices
        
        input_0_values = tl.load(in_ptr0 + (reduction_2d_indices + 4 * input_3d_indices * kernel_size * kernel_size), reduction_mask & input_mask, eviction_policy='evict_first', other=0.0)
        combined_values = input_0_values + input_1_values
        scaling_factor = 0.5
        scaled_values = combined_values * scaling_factor
        broadcasted_values = tl.broadcast_to(scaled_values, [XBLOCK, RBLOCK])
        updated_accumulated_result = accumulated_result + broadcasted_values
        accumulated_result = tl.where(reduction_mask & input_mask, updated_accumulated_result, accumulated_result)
    
    summed_result = tl.sum(accumulated_result, 1)[:, None]
    kernel_area = 4 * kernel_size * kernel_size
    kernel_area_float = kernel_area.to(tl.float32)
    averaged_result = summed_result / kernel_area_float
    normalization_factor = 1.0
    final_result = averaged_result / normalization_factor
    
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (input_3d_indices), final_result, input_mask)