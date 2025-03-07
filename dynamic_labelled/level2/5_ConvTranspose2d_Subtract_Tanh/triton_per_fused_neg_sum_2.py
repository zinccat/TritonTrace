# From: 5_ConvTranspose2d_Subtract_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_neg_sum_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    
    # Calculate the starting index for the input data
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < xnumel
    
    # Calculate the range of output indices
    output_indices = tl.arange(0, RBLOCK)[None, :]
    output_mask = output_indices < rnumel
    
    # Load input data with masking
    input_data = tl.load(in_ptr0 + (input_indices + 16 * output_indices), output_mask & input_mask, other=0.0)
    
    # Broadcast input data to match the output block size
    broadcasted_data = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
    
    # Apply mask and zero out elements outside the mask
    masked_data = tl.where(output_mask & input_mask, broadcasted_data, 0)
    
    # Sum along the output block dimension
    summed_data = tl.sum(masked_data, 1)[:, None]
    
    # Store the result back to the output pointer
    tl.store(out_ptr0 + (input_indices), summed_data, input_mask)