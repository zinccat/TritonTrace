# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_mul_sum_6(input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_num_elements = 240
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_index // 16
    x0 = (x_index % 16)
    temp_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_index

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index

        divisor = triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1), 15)
        tmp0 = r2 + x1 * divisor
        tmp1 = 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1)
        tmp2 = tmp0 < tmp1

        load_index0 = (((-2) * (((r2 + x1 * divisor) // kernel_size3) % kernel_size3)) + 
                       4 * x0 + 
                       64 * (((r2 + x1 * divisor) // kernel_size2) % kernel_size0) + 
                       kernel_size1 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size3) + 
                       x0 * kernel_size1 * kernel_size1 + 
                       ((-64) * kernel_size1 * (((r2 + x1 * divisor) // kernel_size2) % kernel_size0)) + 
                       ((-4) * kernel_size1 * x0) + 
                       16 * kernel_size1 * kernel_size1 * (((r2 + x1 * divisor) // kernel_size2) % kernel_size0) + 
                       ((r2 + x1 * divisor) % kernel_size3))

        tmp3 = tl.load(input_ptr0 + load_index0, rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(input_ptr1 + load_index0, rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(input_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)

        tmp6 = tmp4 + tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = temp_result + tmp10
        temp_result = tl.where(rmask & xmask, tmp12, temp_result)

    temp_sum = tl.sum(temp_result, 1)[:, None]
    tl.store(output_ptr0 + (x3), temp_sum, xmask)