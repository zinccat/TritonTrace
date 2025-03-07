# From: 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sum_0red_fused_mul_sum_0(input_ptr0, input_ptr1, output_ptr0, num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    num_elements_x = 16
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
        temp_input0 = tl.load(input_ptr0 + (r_indices_1), r_mask, eviction_policy='evict_last', other=0.0)
        temp_input1 = tl.load(input_ptr1 + (x_indices_0 + 16 * r_indices_1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        scalar_multiplier = 10.0
        temp_product0 = temp_input0 * scalar_multiplier
        temp_product1 = temp_product0 * temp_input1
        temp_broadcasted = tl.broadcast_to(temp_product1, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcasted
        temp_accumulator = tl.where(r_mask & x_mask, temp_sum, temp_accumulator)
    
    temp_sum_reduced = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (x_indices_0), temp_sum_reduced, x_mask)