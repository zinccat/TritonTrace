# From: 44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_mean_mul_0red_fused_convolution_mean_mul_0(
    output_ptr, input_ptr0, input_ptr1, kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_3d = x_indices
    x_indices_0d = (x_indices % 16)
    
    input_values_1 = tl.load(input_ptr1 + (x_indices_0d), x_mask, eviction_policy='evict_last')
    accumulated_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_2d = r_indices
        
        input_values_0 = tl.load(input_ptr0 + (r_indices_2d + 4 * x_indices_3d * kernel_size * kernel_size), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        combined_values = input_values_0 + input_values_1
        scaling_factor = 0.5
        scaled_values = combined_values * scaling_factor
        broadcasted_values = tl.broadcast_to(scaled_values, [XBLOCK, RBLOCK])
        updated_accumulated_result = accumulated_result + broadcasted_values
        accumulated_result = tl.where(r_mask & x_mask, updated_accumulated_result, accumulated_result)
    
    summed_result = tl.sum(accumulated_result, 1)[:, None]
    kernel_area = 4 * kernel_size * kernel_size
    kernel_area_float = kernel_area.to(tl.float32)
    averaged_result = summed_result / kernel_area_float
    normalization_factor = 1.0
    final_result = averaged_result / normalization_factor
    
    tl.debug_barrier()
    tl.store(output_ptr + (x_indices_3d), final_result, x_mask)