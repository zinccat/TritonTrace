# From: 99_TripletMarginLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_add_clamp_min_mean_norm_sub_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_indices = tl.arange(0, RBLOCK)[None, :]
    
    tmp0 = tl.load(in_ptr0 + (r_indices), None)
    tmp4 = tl.load(in_ptr1 + (r_indices), None)
    
    sqrt_tmp0 = tl.extra.cuda.libdevice.sqrt(tmp0)
    one = 1.0
    tmp3 = sqrt_tmp0 + one
    
    sqrt_tmp4 = tl.extra.cuda.libdevice.sqrt(tmp4)
    tmp6 = tmp3 - sqrt_tmp4
    
    zero = 0.0
    clamped_values = triton_helpers.maximum(tmp6, zero)
    
    broadcast_clamped = tl.broadcast_to(clamped_values, [XBLOCK, RBLOCK])
    sum_clamped = tl.sum(broadcast_clamped, 1)[:, None]
    
    divisor = 128.0
    mean_clamped = sum_clamped / divisor
    
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), mean_clamped, None)