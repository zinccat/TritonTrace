# From: 76_Gemm_Add_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_threshold_backward_0red_fused_sum_threshold_backward_0(
    input_ptr0, input_ptr1, output_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_1 = r_indices
        temp_mask = tl.load(
            input_ptr0 + (x_indices_0 + 512 * r_indices_1),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        ).to(tl.int1)
        
        temp_values = tl.load(
            input_ptr1 + (x_indices_0 + 512 * r_indices_1),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        
        temp_zero = 0.0
        temp_selected = tl.where(temp_mask, temp_zero, temp_values)
        temp_broadcast = tl.broadcast_to(temp_selected, [XBLOCK, RBLOCK])
        temp_accumulated = temp_sum + temp_broadcast
        temp_sum = tl.where(r_mask & x_mask, temp_accumulated, temp_sum)
    
    temp_result = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (x_indices_0), temp_result, x_mask)