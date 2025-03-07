# From: 33_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sum_2(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    accumulated_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_1 = r_indices
        input0_values = tl.load(in_ptr0 + (x_indices_0 + 512 * r_indices_1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input1_values = tl.load(in_ptr1 + (x_indices_0 + 512 * r_indices_1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        multiplied_values = input0_values * input1_values
        broadcasted_values = tl.broadcast_to(multiplied_values, [XBLOCK, RBLOCK])
        updated_accumulated_result = accumulated_result + broadcasted_values
        accumulated_result = tl.where(r_mask & x_mask, updated_accumulated_result, accumulated_result)
    
    summed_result = tl.sum(accumulated_result, 1)[:, None]
    tl.store(out_ptr0 + (x_indices_0), summed_result, x_mask)