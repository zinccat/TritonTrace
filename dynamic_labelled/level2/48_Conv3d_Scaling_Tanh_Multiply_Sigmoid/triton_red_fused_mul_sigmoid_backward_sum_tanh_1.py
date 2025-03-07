# From: 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sigmoid_backward_sum_tanh_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, output_ptr1, kernel_size0, kernel_size1, kernel_size2, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 336
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_index // 16
    x0 = x_index % 16
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_index
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index

        divisor = triton_helpers.div_floor_integer(
            20 + ((-8) * kernel_size0) + ((-2) * kernel_size0 * kernel_size2 * kernel_size2) + 
            4 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size2 + 
            kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2 + ((-4) * kernel_size0 * kernel_size1 * kernel_size2), 
            21
        )

        tmp0 = r2 + x1 * divisor
        tmp1 = ((-8) * kernel_size0) + ((-2) * kernel_size0 * kernel_size2 * kernel_size2) + \
               4 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size2 + \
               kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2 + ((-4) * kernel_size0 * kernel_size1 * kernel_size2)
        tmp2 = tmp0 < tmp1

        index_expr = (
            (-128) * (((r2 + x1 * divisor) // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
            8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)
            + ((-8) * x0) + ((-2) * (((r2 + x1 * divisor) // ((-2) + kernel_size2)) % ((-2) + kernel_size2)))
            + 4 * (((r2 + x1 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1))
            + kernel_size2 * (((r2 + x1 * divisor) // ((-2) + kernel_size2)) % ((-2) + kernel_size2))
            + kernel_size2 * kernel_size2 * (((r2 + x1 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1))
            + ((-32) * kernel_size2 * kernel_size2 * (((r2 + x1 * divisor) // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
            8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))
            + ((-4) * kernel_size2 * (((r2 + x1 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1)))
            + ((-2) * x0 * kernel_size2 * kernel_size2) + 4 * kernel_size1 * x0 + 8 * kernel_size2 * x0
            + 64 * kernel_size1 * (((r2 + x1 * divisor) // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
            8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)
            + 128 * kernel_size2 * (((r2 + x1 * divisor) // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
            8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)
            + kernel_size1 * x0 * kernel_size2 * kernel_size2
            + ((-64) * kernel_size1 * kernel_size2 * (((r2 + x1 * divisor) // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
            8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))
            + ((-4) * kernel_size1 * kernel_size2 * x0)
            + 16 * kernel_size1 * kernel_size2 * kernel_size2 * (((r2 + x1 * divisor) // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
            8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)
            + ((r2 + x1 * divisor) % ((-2) + kernel_size2))
        )

        tmp3 = tl.load(input_ptr0 + index_expr, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(input_ptr1 + index_expr, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = 1.0
        tmp6 = tmp5 - tmp4
        tmp7 = tmp4 * tmp6
        tmp8 = tmp3 * tmp7
        tmp9 = tl.load(input_ptr2 + index_expr, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(input_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 * tmp10
        tmp12 = tl.extra.cuda.libdevice.tanh(tmp11)
        tmp13 = tmp8 * tmp12
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r_mask & x_mask, tmp18, _tmp17)
        tmp19 = tl.load(input_ptr4 + index_expr, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp19 * tmp9
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(r_mask & x_mask, tmp25, _tmp24)

    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(output_ptr0 + (x3), tmp17, x_mask)
    tl.store(output_ptr1 + (x3), tmp24, x_mask)