# From: 100_HingeLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_clamp_mean_mul_rsub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_indices = tl.arange(0, RBLOCK)[None, :]
    
    # Load data from input pointers
    input_data0 = tl.load(in_ptr0 + (r_indices), None)
    input_data1 = tl.load(in_ptr1 + (r_indices), None)
    
    # Element-wise multiplication
    multiplied_data = input_data0 * input_data1
    
    # Compute 1.0 - multiplied_data
    one_minus_multiplied = 1.0 - multiplied_data
    
    # Clamp the result to be non-negative
    clamped_data = triton_helpers.maximum(one_minus_multiplied, 0.0)
    
    # Broadcast the clamped data
    broadcast_clamped = tl.broadcast_to(clamped_data, [XBLOCK, RBLOCK])
    
    # Sum along the second dimension
    summed_data = tl.sum(broadcast_clamped, 1)[:, None]
    
    # Compute the mean by dividing by RBLOCK
    mean_data = summed_data / 128.0
    
    # Store the result
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), mean_data, None)