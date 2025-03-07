# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mul_native_group_norm_sum_3(input_ptr, output_ptr, num_elements, reduced_num_elements, XBLOCK: tl.constexpr):
    num_elements = 16
    reduced_num_elements = 15
    RBLOCK: tl.constexpr = 16
    
    # Calculate the offset for the current program ID
    element_offset = tl.program_id(0) * XBLOCK
    
    # Generate indices for the input elements
    element_indices = element_offset + tl.arange(0, XBLOCK)[:, None]
    element_mask = element_indices < num_elements
    
    # Generate indices for the reduced elements
    reduced_indices = tl.arange(0, RBLOCK)[None, :]
    reduced_mask = reduced_indices < reduced_num_elements
    
    # Load data from input pointer with masking
    loaded_data = tl.load(input_ptr + (element_indices + 16 * reduced_indices), reduced_mask & element_mask, other=0.0)
    
    # Broadcast loaded data to match the block size
    broadcasted_data = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])
    
    # Apply mask and zero out elements outside the mask
    masked_data = tl.where(reduced_mask & element_mask, broadcasted_data, 0)
    
    # Sum along the reduced dimension
    summed_data = tl.sum(masked_data, 1)[:, None]
    
    # Store the result in the output pointer with masking
    tl.store(output_ptr + (element_indices), summed_data, element_mask)