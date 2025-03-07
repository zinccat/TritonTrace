# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_mul_sum_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, kernel_size2, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 310
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_indices
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r1 = r_indices

        divisor = triton_helpers.div_floor_integer(
            309 + ((-16) * kernel_size0) + ((-16) * kernel_size0 * kernel_size2 * kernel_size2) + 
            16 * kernel_size0 * kernel_size1 + 32 * kernel_size0 * kernel_size2 + 
            ((-32) * kernel_size0 * kernel_size1 * kernel_size2) + 16 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
            310
        )

        tmp0 = r1 + x0 * divisor
        tmp1 = ((-16) * kernel_size0) + ((-16) * kernel_size0 * kernel_size2 * kernel_size2) + 
               16 * kernel_size0 * kernel_size1 + 32 * kernel_size0 * kernel_size2 + 
               ((-32) * kernel_size0 * kernel_size1 * kernel_size2) + 16 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2
        tmp2 = tmp0 < tmp1

        index_expr = (
            (-1) * (((r1 + x0 * divisor) // ((-1) + kernel_size2)) % ((-1) + kernel_size2)) + 
            (-1) * (((r1 + x0 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + 
            kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % (16 * kernel_size0)) + 
            kernel_size1 * (((r1 + x0 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + 
            kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % (16 * kernel_size0)) + 
            kernel_size2 * (((r1 + x0 * divisor) // ((-1) + kernel_size2)) % ((-1) + kernel_size2)) + 
            kernel_size2 * kernel_size2 * (((r1 + x0 * divisor) // (1 + kernel_size2 * kernel_size2 + ((-2) * kernel_size2))) % ((-1) + kernel_size1)) + 
            ((-1) * kernel_size2 * kernel_size2 * (((r1 + x0 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + 
            kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % (16 * kernel_size0))) + 
            ((-2) * kernel_size2 * (((r1 + x0 * divisor) // (1 + kernel_size2 * kernel_size2 + ((-2) * kernel_size2))) % ((-1) + kernel_size1))) + 
            2 * kernel_size2 * (((r1 + x0 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + 
            kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % (16 * kernel_size0)) + 
            kernel_size1 * kernel_size2 * kernel_size2 * (((r1 + x0 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + 
            kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % (16 * kernel_size0)) + 
            ((-2) * kernel_size1 * kernel_size2 * (((r1 + x0 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + 
            kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % (16 * kernel_size0))) + 
            ((r1 + x0 * divisor) % ((-1) + kernel_size2)) + (((r1 + x0 * divisor) // (1 + kernel_size2 * kernel_size2 + ((-2) * kernel_size2))) % ((-1) + kernel_size1))
        )

        tmp3 = tl.load(input_ptr0 + index_expr, rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(input_ptr1 + index_expr, rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(input_ptr2 + (((r1 + x0 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + 
            kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % 16), rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)

        tmp6 = tmp4 + tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = temp_accumulator + tmp10
        temp_accumulator = tl.where(rmask & x_mask, tmp12, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (x0), temp_sum, x_mask)