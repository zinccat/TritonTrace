# From: 17_Conv2d_InstanceNorm_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    
    # Calculate the starting index for the current program
    x_start_index = tl.program_id(0) * XBLOCK
    x_indices = x_start_index + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    
    # Load data from input pointer with masking
    input_data = tl.load(in_ptr0 + (x_indices + 16 * r_indices), r_mask & x_mask, other=0.0)
    
    # Broadcast loaded data to match dimensions
    broadcasted_data = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
    
    # Apply mask and zero out elements outside the mask
    masked_data = tl.where(r_mask & x_mask, broadcasted_data, 0)
    
    # Sum along the second dimension and reshape
    summed_data = tl.sum(masked_data, 1)[:, None]
    
    # Store the result in the output pointer with masking
    tl.store(out_ptr0 + (x_indices), summed_data, x_mask)