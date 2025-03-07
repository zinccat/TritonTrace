# From: 76_Gemm_Add_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_threshold_backward_0(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    accumulated_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_flat = r_indices
        input_mask = tl.load(in_ptr0 + (x_indices_flat + 512 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        input_values = tl.load(in_ptr1 + (x_indices_flat + 512 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        zero_values = 0.0
        masked_values = tl.where(input_mask, zero_values, input_values)
        broadcasted_values = tl.broadcast_to(masked_values, [XBLOCK, RBLOCK])
        updated_sum = accumulated_sum + broadcasted_values
        accumulated_sum = tl.where(r_mask & x_mask, updated_sum, accumulated_sum)
    
    reduced_sum = tl.sum(accumulated_sum, 1)[:, None]
    tl.store(out_ptr0 + (x_indices_flat), reduced_sum, x_mask)