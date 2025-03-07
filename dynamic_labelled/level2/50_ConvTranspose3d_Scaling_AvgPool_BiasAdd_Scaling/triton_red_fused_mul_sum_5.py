# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sum_5red_fused_mul_sum_5(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, kernel_size4, kernel_size5,
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_num_elements = 1923
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    _accumulated_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r1 = r_index

        divisor = triton_helpers.div_floor_integer(
            1922 + ((-16) * kernel_size0) + ((-64) * kernel_size0 * kernel_size2 * kernel_size2) +
            32 * kernel_size0 * kernel_size1 + 64 * kernel_size0 * kernel_size2 +
            ((-128) * kernel_size0 * kernel_size1 * kernel_size2) + 128 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2,
            1923
        )

        tmp0 = r1 + x0 * divisor
        tmp1 = ((-16) * kernel_size0) + ((-64) * kernel_size0 * kernel_size2 * kernel_size2) + 32 * kernel_size0 * kernel_size1 + 64 * kernel_size0 * kernel_size2 + ((-128) * kernel_size0 * kernel_size1 * kernel_size2) + 128 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2
        tmp2 = tmp0 < tmp1

        index0 = (-1) * (((r1 + x0 * divisor) // kernel_size3) % kernel_size3)
        index1 = (-1) * (((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size2 * kernel_size2) + 2 * kernel_size1 + 4 * kernel_size2 + ((-8) * kernel_size1 * kernel_size2) + 8 * kernel_size1 * kernel_size2 * kernel_size2)) % (16 * kernel_size0))
        index2 = (-4) * kernel_size2 * (((r1 + x0 * divisor) // kernel_size5) % kernel_size4)
        index3 = (-4) * kernel_size2 * kernel_size2 * (((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size2 * kernel_size2) + 2 * kernel_size1 + 4 * kernel_size2 + ((-8) * kernel_size1 * kernel_size2) + 8 * kernel_size1 * kernel_size2 * kernel_size2)) % (16 * kernel_size0))
        index4 = 2 * kernel_size1 * (((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size2 * kernel_size2) + 2 * kernel_size1 + 4 * kernel_size2 + ((-8) * kernel_size1 * kernel_size2) + 8 * kernel_size1 * kernel_size2 * kernel_size2)) % (16 * kernel_size0))
        index5 = 2 * kernel_size2 * (((r1 + x0 * divisor) // kernel_size3) % kernel_size3)
        index6 = 4 * kernel_size2 * (((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size2 * kernel_size2) + 2 * kernel_size1 + 4 * kernel_size2 + ((-8) * kernel_size1 * kernel_size2) + 8 * kernel_size1 * kernel_size2 * kernel_size2)) % (16 * kernel_size0))
        index7 = 4 * kernel_size2 * kernel_size2 * (((r1 + x0 * divisor) // kernel_size5) % kernel_size4)
        index8 = (-8) * kernel_size1 * kernel_size2 * (((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size2 * kernel_size2) + 2 * kernel_size1 + 4 * kernel_size2 + ((-8) * kernel_size1 * kernel_size2) + 8 * kernel_size1 * kernel_size2 * kernel_size2)) % (16 * kernel_size0))
        index9 = 8 * kernel_size1 * kernel_size2 * kernel_size2 * (((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size2 * kernel_size2) + 2 * kernel_size1 + 4 * kernel_size2 + ((-8) * kernel_size1 * kernel_size2) + 8 * kernel_size1 * kernel_size2 * kernel_size2)) % (16 * kernel_size0))
        index10 = ((r1 + x0 * divisor) % kernel_size3)
        index11 = (((r1 + x0 * divisor) // kernel_size5) % kernel_size4)

        load_index0 = index0 + index1 + index2 + index3 + index4 + index5 + index6 + index7 + index8 + index9 + index10 + index11

        tmp3 = tl.load(input_ptr0 + load_index0, rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(input_ptr1 + load_index0, rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)

        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _accumulated_sum + tmp8
        _accumulated_sum = tl.where(rmask & x_mask, tmp10, _accumulated_sum)

    tmp9 = tl.sum(_accumulated_sum, 1)[:, None]
    tl.store(output_ptr0 + (x0), tmp9, x_mask)