# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_mean_native_group_norm_2(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 4
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    loaded_values = tl.load(in_ptr0 + (r1 + (4 * x0)), x_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(x_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]
    normalization_factor = 201600.0
    mean_values = summed_values / normalization_factor
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), mean_values, x_mask)