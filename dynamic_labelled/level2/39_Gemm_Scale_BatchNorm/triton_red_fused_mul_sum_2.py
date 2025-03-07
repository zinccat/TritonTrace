# From: 39_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sum_2red_fused_mul_sum_2(input_ptr0, input_ptr1, output_ptr0, num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    num_elements_x = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_indices_1 = r_indices
        temp0 = tl.load(input_ptr0 + (x_indices_0 + 512 * r_indices_1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        temp1 = tl.load(input_ptr1 + (x_indices_0 + 512 * r_indices_1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        temp2 = temp0 * temp1
        temp3 = tl.broadcast_to(temp2, [XBLOCK, RBLOCK])
        temp5 = temp_accumulator + temp3
        temp_accumulator = tl.where(r_mask & x_mask, temp5, temp_accumulator)
    
    temp4 = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (x_indices_0), temp4, x_mask)