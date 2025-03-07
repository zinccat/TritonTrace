# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_mul_sum_6red_fused_add_mul_sum_6(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_num_elements = 240
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_index // 16
    x0 = x_index % 16
    temp_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_index

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index

        divisor = triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
        tmp0 = r2 + x1 * divisor
        tmp1 = 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1
        tmp2 = tmp0 < tmp1

        load_index0 = (
            (-2) * (((r2 + x1 * divisor) // kernel_size3) % kernel_size3) 
            + 4 * x0 
            + 64 * (((r2 + x1 * divisor) // kernel_size2) % kernel_size0) 
            + kernel_size1 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size3) 
            + x0 * kernel_size1 * kernel_size1 
            + (-64) * kernel_size1 * (((r2 + x1 * divisor) // kernel_size2) % kernel_size0) 
            + (-4) * kernel_size1 * x0 
            + 16 * kernel_size1 * kernel_size1 * (((r2 + x1 * divisor) // kernel_size2) % kernel_size0) 
            + ((r2 + x1 * divisor) % kernel_size3)
        )

        load_index1 = load_index0
        load_index2 = tl.broadcast_to(x0, [XBLOCK, RBLOCK])

        input_value0 = tl.load(input_ptr0 + load_index0, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        input_value1 = tl.load(input_ptr1 + load_index1, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        input_value2 = tl.load(input_ptr2 + load_index2, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)

        temp_sum = input_value1 + input_value2
        temp_product = input_value0 * temp_sum

        temp_broadcast = tl.full(temp_product.shape, 0, temp_product.dtype)
        temp_conditional = tl.where(tmp2, temp_product, temp_broadcast)
        temp_broadcasted = tl.broadcast_to(temp_conditional, [XBLOCK, RBLOCK])

        temp_result = temp_result + temp_broadcasted
        temp_result = tl.where(r_mask & x_mask, temp_result, temp_result)

    final_sum = tl.sum(temp_result, 1)[:, None]
    tl.store(output_ptr0 + (x3), final_sum, x_mask)