# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_div_ge_le_logical_and_logsumexp_mul_neg_sigmoid_sub_sum_where_3(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_num_elements = 1968
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_index // 16
    x0 = x_index % 16
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_index

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index

        divisor = triton_helpers.div_floor_integer(
            122 + ((-1) * kernel_size0) + ((-4) * kernel_size0 * kernel_size2 * kernel_size2) + 
            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
            ((-8) * kernel_size0 * kernel_size1 * kernel_size2) + 8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
            123
        )

        tmp0 = r2 + x1 * divisor
        tmp1 = ((-1) * kernel_size0) + ((-4) * kernel_size0 * kernel_size2 * kernel_size2) + 
               2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
               ((-8) * kernel_size0 * kernel_size1 * kernel_size2) + 8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2
        tmp2 = tmp0 < tmp1

        load_index0 = (
            (-1) * (
                (((r2 + x1 * divisor) // kernel_size3) % kernel_size0)
                + ((-1) * (((r2 + x1 * divisor) // ((-1) + 2 * kernel_size2)) % ((-1) + 2 * kernel_size2)))
                + ((-4) * kernel_size2 * (((r2 + x1 * divisor) // (1 + ((-4) * kernel_size2) + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1)))
                + ((-4) * kernel_size2 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0))
                + 2 * kernel_size1 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0)
                + 2 * kernel_size2 * (((r2 + x1 * divisor) // ((-1) + 2 * kernel_size2)) % ((-1) + 2 * kernel_size2))
                + 4 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0)
                + 4 * kernel_size2 * kernel_size2 * (((r2 + x1 * divisor) // (1 + ((-4) * kernel_size2) + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1))
                + ((-8) * kernel_size1 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0))
                + 8 * kernel_size1 * kernel_size2 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0)
                + (((r2 + x1 * divisor) % ((-1) + 2 * kernel_size2)))
                + ((((r2 + x1 * divisor) // (1 + ((-4) * kernel_size2) + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1)))
            )
        )

        tmp3 = tl.load(input_ptr0 + load_index0, rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = 3.0
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sigmoid(tmp5)
        tmp7 = tmp3 * tmp6
        tmp8 = 0.16666666666666666
        tmp9 = tmp7 * tmp8

        tmp10 = tl.load(input_ptr1 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = -1.0
        tmp13 = tmp11 >= tmp12
        tmp14 = 1.0
        tmp15 = tmp11 <= tmp14
        tmp16 = tmp13 & tmp15

        load_index2 = (
            (-1) * x0
            + ((-1) * (((r2 + x1 * divisor) // ((-1) + 2 * kernel_size2)) % ((-1) + 2 * kernel_size2)))
            + ((-16) * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0))
            + ((-64) * kernel_size2 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0))
            + ((-4) * kernel_size2 * (((r2 + x1 * divisor) // (1 + ((-4) * kernel_size2) + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1)))
            + ((-4) * x0 * kernel_size2 * kernel_size2)
            + 2 * kernel_size1 * x0
            + 2 * kernel_size2 * (((r2 + x1 * divisor) // ((-1) + 2 * kernel_size2)) % ((-1) + 2 * kernel_size2))
            + 4 * kernel_size2 * x0
            + 4 * kernel_size2 * kernel_size2 * (((r2 + x1 * divisor) // (1 + ((-4) * kernel_size2) + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1))
            + 32 * kernel_size1 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0)
            + 64 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0)
            + ((-128) * kernel_size1 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0))
            + ((-8) * kernel_size1 * kernel_size2 * x0)
            + 8 * kernel_size1 * x0 * kernel_size2 * kernel_size2
            + 128 * kernel_size1 * kernel_size2 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0)
            + (((r2 + x1 * divisor) % ((-1) + 2 * kernel_size2)))
            + ((((r2 + x1 * divisor) // (1 + ((-4) * kernel_size2) + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1)))
        )

        tmp17 = tl.load(input_ptr2 + load_index2, rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = 0.0
        tmp19 = tl.where(tmp16, tmp17, tmp18)
        tmp20 = -tmp19
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        temp_accumulator = temp_accumulator + tmp23
        temp_accumulator = tl.where(rmask & x_mask, temp_accumulator, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (x3), temp_result, x_mask)