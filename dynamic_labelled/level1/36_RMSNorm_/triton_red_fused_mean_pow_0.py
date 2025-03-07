# From: 36_RMSNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_pow_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = (x_indices % ks0)
    x1 = x_indices // ks0
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_indices
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r2 = r_indices
        loaded_values = tl.load(in_ptr0 + (x0 + r2 * ks2 * ks2 + ks1 * x1 * ks2 * ks2), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        squared_values = loaded_values * loaded_values
        broadcasted_values = tl.broadcast_to(squared_values, [XBLOCK, RBLOCK])
        temp_sum_update = temp_sum + broadcasted_values
        temp_sum = tl.where(r_mask & x_mask, temp_sum_update, temp_sum)
    summed_values = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (x3), summed_values, x_mask)