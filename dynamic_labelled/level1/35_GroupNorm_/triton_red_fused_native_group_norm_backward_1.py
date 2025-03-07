# From: 35_GroupNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_1(in_ptr0, out_ptr0, kernel_size, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_flat = r_indices
        loaded_values = tl.load(in_ptr0 + (r_indices_flat + x_indices_flat * kernel_size * kernel_size), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_accumulated = temp_sum + broadcasted_values
        temp_sum = tl.where(r_mask & x_mask, temp_accumulated, temp_sum)
    
    summed_values = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (x_indices_flat), summed_values, x_mask)