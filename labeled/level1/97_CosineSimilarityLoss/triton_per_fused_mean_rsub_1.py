# From: 97_CosineSimilarityLoss

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_mean_rsub_1(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_indices = tl.arange(0, RBLOCK)[None, :]
    
    # Load the input data
    loaded_values = tl.load(in_ptr0 + (r_indices), None)
    
    # Compute the subtraction
    subtracted_values = 1.0 - loaded_values
    
    # Broadcast the subtracted values
    broadcasted_values = tl.broadcast_to(subtracted_values, [XBLOCK, RBLOCK])
    
    # Sum along the rows
    summed_values = tl.sum(broadcasted_values, 1)[:, None]
    
    # Compute the mean
    mean_values = summed_values / 128.0
    
    # Store the result
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), mean_values, None)