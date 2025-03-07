# From: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_backward_data_mul_sum_tanh_tanh_backward_0(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 241
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r1 = r_indices

        divisor = triton_helpers.div_floor_integer(
            240 + ((-1) * kernel_size0) + 2 * kernel_size0 * kernel_size0 + 
            ((-8) * kernel_size1 * kernel_size0 * kernel_size0) + 
            ((-4) * kernel_size0 * kernel_size1 * kernel_size1) + 
            4 * kernel_size0 * kernel_size1 + 
            8 * kernel_size0 * kernel_size0 * kernel_size1 * kernel_size1, 
            241
        )

        tmp0 = r1 + x0 * divisor
        tmp1 = ((-1) * kernel_size0) + 2 * kernel_size0 * kernel_size0 + \
               ((-8) * kernel_size1 * kernel_size0 * kernel_size0) + \
               ((-4) * kernel_size0 * kernel_size1 * kernel_size1) + \
               4 * kernel_size0 * kernel_size1 + \
               8 * kernel_size0 * kernel_size0 * kernel_size1 * kernel_size1

        tmp2 = tmp0 < tmp1

        index0 = (((-1) * (((r1 + x0 * divisor) // ((-1) + 2 * kernel_size1)) % ((-1) + 2 * kernel_size1)))) + \
                 ((-1) * (((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size1 * kernel_size1) + 2 * kernel_size0 + 4 * kernel_size1 + ((-8) * kernel_size0 * kernel_size1) + 8 * kernel_size0 * kernel_size1 * kernel_size1)) % kernel_size0))) + \
                 ((-4) * kernel_size1 * ((((r1 + x0 * divisor) // (1 + ((-4) * kernel_size1) + 4 * kernel_size1 * kernel_size1)) % ((-1) + 2 * kernel_size0)) % ((-1) + 2 * kernel_size0))) + \
                 ((-4) * kernel_size1 * kernel_size1 * ((((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size1 * kernel_size1) + 2 * kernel_size0 + 4 * kernel_size1 + ((-8) * kernel_size0 * kernel_size1) + 8 * kernel_size0 * kernel_size1 * kernel_size1)) % kernel_size0)) % kernel_size0)) + \
                 2 * kernel_size0 * ((((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size1 * kernel_size1) + 2 * kernel_size0 + 4 * kernel_size1 + ((-8) * kernel_size0 * kernel_size1) + 8 * kernel_size0 * kernel_size1 * kernel_size1)) % kernel_size0)) + \
                 2 * kernel_size1 * ((((r1 + x0 * divisor) // ((-1) + 2 * kernel_size1)) % ((-1) + 2 * kernel_size1)) % ((-1) + 2 * kernel_size1)) + \
                 4 * kernel_size1 * ((((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size1 * kernel_size1) + 2 * kernel_size0 + 4 * kernel_size1 + ((-8) * kernel_size0 * kernel_size1) + 8 * kernel_size0 * kernel_size1 * kernel_size1)) % kernel_size0)) + \
                 4 * kernel_size1 * kernel_size1 * ((((r1 + x0 * divisor) // (1 + ((-4) * kernel_size1) + 4 * kernel_size1 * kernel_size1)) % ((-1) + 2 * kernel_size0)) % ((-1) + 2 * kernel_size0)) + \
                 ((-8) * kernel_size0 * kernel_size1 * ((((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size1 * kernel_size1) + 2 * kernel_size0 + 4 * kernel_size1 + ((-8) * kernel_size0 * kernel_size1) + 8 * kernel_size0 * kernel_size1 * kernel_size1)) % kernel_size0)) % kernel_size0)) + \
                 8 * kernel_size0 * kernel_size1 * kernel_size1 * ((((r1 + x0 * divisor) // ((-1) + ((-4) * kernel_size1 * kernel_size1) + 2 * kernel_size0 + 4 * kernel_size1 + ((-8) * kernel_size0 * kernel_size1) + 8 * kernel_size0 * kernel_size1 * kernel_size1)) % kernel_size0)) % kernel_size0) + \
                 (((r1 + x0 * divisor) % ((-1) + 2 * kernel_size1))) + \
                 ((((r1 + x0 * divisor) // (1 + ((-4) * kernel_size1) + 4 * kernel_size1 * kernel_size1)) % ((-1) + 2 * kernel_size0))))

        tmp3 = tl.load(input_ptr0 + index0, rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = -tmp3

        tmp5 = tl.load(input_ptr1 + index0, rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = 2.0
        tmp7 = tmp5 * tmp6
        tmp8 = tl.extra.cuda.libdevice.tanh(tmp3)
        tmp9 = tmp8 * tmp8
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp7 * tmp11
        tmp13 = tmp12 * tmp3
        tmp14 = tl.extra.cuda.libdevice.fma(tmp4, tmp13, tmp13)
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        temp_sum = temp_sum + tmp17

        temp_sum = tl.where(rmask & x_mask, temp_sum, temp_sum)

    temp_result = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (x0), temp_result, x_mask)