# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

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
        temp_load = tl.load(
            in_ptr0 + (r_indices_flat + 4 * x_indices_flat + x_indices_flat * kernel_size * kernel_size + ((-4) * kernel_size * x_indices_flat)),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_accumulate = temp_sum + temp_broadcast
        temp_sum = tl.where(r_mask & x_mask, temp_accumulate, temp_sum)
    
    temp_result = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (x_indices_flat), temp_result, x_mask)