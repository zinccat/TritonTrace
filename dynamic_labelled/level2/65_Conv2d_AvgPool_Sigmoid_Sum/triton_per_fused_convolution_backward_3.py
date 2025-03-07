# From: 65_Conv2d_AvgPool_Sigmoid_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    
    # Calculate the starting index for the input data
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < xnumel
    
    # Calculate the range of output indices
    output_indices = tl.arange(0, RBLOCK)[None, :]
    output_mask = output_indices < rnumel
    
    # Load input data with masking
    input_row_indices = output_indices
    input_col_indices = input_indices
    loaded_data = tl.load(in_ptr0 + (input_col_indices + 16 * input_row_indices), output_mask & input_mask, other=0.0)
    
    # Broadcast loaded data to match dimensions
    broadcasted_data = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])
    
    # Apply mask and zero out where necessary
    masked_data = tl.where(output_mask & input_mask, broadcasted_data, 0)
    
    # Sum along the output dimension
    summed_data = tl.sum(masked_data, 1)[:, None]
    
    # Store the result back to the output pointer
    tl.store(out_ptr0 + (input_col_indices), summed_data, input_mask)