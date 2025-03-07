# From: 57_Conv2d_ReLU_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    
    # Calculate the starting index for the input data
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    
    # Create a mask to ensure indices are within bounds
    input_mask = input_indices < xnumel
    reduction_indices = tl.arange(0, RBLOCK)[None, :]
    reduction_mask = reduction_indices < rnumel
    
    # Load data from input pointer with masking
    loaded_data = tl.load(in_ptr0 + (input_indices + 16 * reduction_indices), reduction_mask & input_mask, other=0.0)
    
    # Broadcast loaded data to match the shape
    broadcasted_data = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])
    
    # Apply mask and zero out out-of-bound elements
    masked_data = tl.where(reduction_mask & input_mask, broadcasted_data, 0)
    
    # Sum along the reduction dimension
    summed_data = tl.sum(masked_data, 1)[:, None]
    
    # Store the result back to the output pointer
    tl.store(out_ptr0 + (input_indices), summed_data, input_mask)