# From: 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_mean_0red_fused_convolution_mean_0(
    input_ptr0, input_ptr1, output_ptr0, kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_3d = x_indices
    x_indices_0d = (x_indices % 16)
    
    input_value1 = tl.load(input_ptr1 + (x_indices_0d), x_mask, eviction_policy='evict_last')
    accumulated_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_2d = r_indices
        
        input_value0 = tl.load(
            input_ptr0 + (r_indices_2d + 4 * x_indices_3d + x_indices_3d * kernel_size * kernel_size + 4 * kernel_size * x_indices_3d),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        
        combined_values = input_value0 + input_value1
        broadcasted_values = tl.broadcast_to(combined_values, [XBLOCK, RBLOCK])
        accumulated_sum += broadcasted_values
        
        accumulated_sum = tl.where(r_mask & x_mask, accumulated_sum, accumulated_sum)
    
    reduced_sum = tl.sum(accumulated_sum, 1)[:, None]
    tl.store(output_ptr0 + (x_indices_3d), reduced_sum, x_mask)