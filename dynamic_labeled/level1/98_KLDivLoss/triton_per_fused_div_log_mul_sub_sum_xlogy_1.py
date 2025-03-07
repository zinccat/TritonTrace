# From: 98_KLDivLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_div_log_mul_sub_sum_xlogy_1(in_out_ptr0, in_ptr0, ks0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_indices = tl.arange(0, RBLOCK)[None, :]
    
    # Load input data
    input_data = tl.load(in_ptr0 + (r_indices), None)
    broadcasted_data = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
    
    # Sum across the second dimension
    summed_data = tl.sum(broadcasted_data, 1)[:, None]
    
    # Convert ks0 to float32
    ks_float = ks0.to(tl.float32)
    
    # Compute division
    result = summed_data / ks_float
    
    # Synchronize threads
    tl.debug_barrier()
    
    # Store the result
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), result, None)