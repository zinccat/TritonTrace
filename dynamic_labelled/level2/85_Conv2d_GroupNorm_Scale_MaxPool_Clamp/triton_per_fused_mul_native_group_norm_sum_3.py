# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mul_native_group_norm_sum_3per_fused_mul_native_group_norm_sum_3(input_ptr, output_ptr, num_elements_x, num_elements_r, XBLOCK: tl.constexpr):
    num_elements_x = 16
    num_elements_r = 15
    RBLOCK: tl.constexpr = 16
    
    # Calculate the offset for the current program ID
    x_offset = tl.program_id(0) * XBLOCK
    
    # Generate indices for x and r dimensions
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_elements_r
    
    # Load data from input pointer with masking
    loaded_data = tl.load(input_ptr + (x_indices + 16 * r_indices), r_mask & x_mask, other=0.0)
    
    # Broadcast loaded data to match dimensions
    broadcasted_data = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])
    
    # Apply mask and zero out where necessary
    masked_data = tl.where(r_mask & x_mask, broadcasted_data, 0)
    
    # Sum along the r dimension
    summed_data = tl.sum(masked_data, 1)[:, None]
    
    # Store the result in the output pointer
    tl.store(output_ptr + (x_indices), summed_data, x_mask)