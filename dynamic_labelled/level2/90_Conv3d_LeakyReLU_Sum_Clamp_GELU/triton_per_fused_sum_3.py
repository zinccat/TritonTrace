# From: 90_Conv3d_LeakyReLU_Sum_Clamp_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_sum_3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    rnumel = 21
    RBLOCK: tl.constexpr = 32
    
    # Calculate the starting index for the current block
    x_start_index = tl.program_id(0) * XBLOCK
    x_indices = x_start_index + tl.arange(0, XBLOCK)[:, None]
    
    # Create a mask to ensure indices are within bounds
    x_within_bounds = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_within_bounds = r_indices < rnumel
    
    # Load data from input pointer with masking
    input_indices = x_indices + 16 * r_indices
    load_mask = r_within_bounds & x_within_bounds
    loaded_data = tl.load(in_ptr0 + input_indices, load_mask, other=0.0)
    
    # Broadcast loaded data to the required shape
    broadcasted_data = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])
    
    # Apply mask and zero out out-of-bounds elements
    masked_data = tl.where(load_mask, broadcasted_data, 0)
    
    # Sum along the second dimension and reshape
    summed_data = tl.sum(masked_data, 1)[:, None]
    
    # Store the result back to the output pointer
    tl.store(out_ptr0 + x_indices, summed_data, x_within_bounds)