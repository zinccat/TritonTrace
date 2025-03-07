# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_native_group_norm_sum_2(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr0, kernel_size0, kernel_size1, kernel_size2, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_num_elements = 240
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_index // 16
    x0 = (x_index % 16)
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_index

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index

        divisor = triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1), 15)
        tmp0 = r2 + x1 * divisor
        tmp1 = 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1)
        tmp2 = tmp0 < tmp1

        index0 = (
            (-2) * (((r2 + x1 * divisor) // kernel_size2) % kernel_size2)
            + 4 * x0
            + 64 * (((r2 + x1 * divisor) // (4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) % kernel_size0)
            + kernel_size1 * (((r2 + x1 * divisor) // kernel_size2) % kernel_size2)
            + x0 * kernel_size1 * kernel_size1
            + (-64) * kernel_size1 * (((r2 + x1 * divisor) // (4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) % kernel_size0)
            + (-4) * kernel_size1 * x0
            + 16 * kernel_size1 * kernel_size1 * (((r2 + x1 * divisor) // (4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) % kernel_size0)
            + ((r2 + x1 * divisor) % kernel_size2)
        )

        input_val0 = tl.load(input_ptr0 + index0, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        input_val1 = tl.load(input_ptr1 + index0, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        input_val2 = tl.load(input_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        input_val3 = tl.load(input_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)

        product = input_val1 * input_val2
        sum_val = product + input_val3
        fused_val = input_val0 * sum_val

        broadcast_fused_val = tl.where(tmp2, fused_val, tl.full(fused_val.shape, 0, fused_val.dtype))
        broadcast_fused_val = tl.broadcast_to(broadcast_fused_val, [XBLOCK, RBLOCK])
        temp_sum = temp_sum + broadcast_fused_val

        temp_sum = tl.where(r_mask & x_mask, temp_sum, temp_sum)

    result_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (x3), result_sum, x_mask)