# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_native_group_norm_sum_2red_fused_mul_native_group_norm_sum_2(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr0, kernel_size0, kernel_size1, kernel_size2, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    x_num_elements = 240
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_index // 16
    x0 = x_index % 16
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
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
            (-2) * (((r2 + x1 * divisor) // kernel_size2) % kernel_size2) 
            + 4 * x0 
            + 64 * (((r2 + x1 * divisor) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0) 
            + kernel_size1 * (((r2 + x1 * divisor) // kernel_size2) % kernel_size2) 
            + x0 * kernel_size1 * kernel_size1 
            + (-64) * kernel_size1 * (((r2 + x1 * divisor) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0) 
            + (-4) * kernel_size1 * x0 
            + 16 * kernel_size1 * kernel_size1 * (((r2 + x1 * divisor) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0) 
            + ((r2 + x1 * divisor) % kernel_size2)
        )

        load_index1 = load_index0
        load_index2 = tl.broadcast_to(x0, [XBLOCK, RBLOCK])
        load_index3 = load_index2

        input_value0 = tl.load(input_ptr0 + load_index0, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        input_value1 = tl.load(input_ptr1 + load_index1, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        input_value2 = tl.load(input_ptr2 + load_index2, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        input_value3 = tl.load(input_ptr3 + load_index3, r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)

        product1 = input_value1 * input_value2
        sum1 = product1 + input_value3
        product2 = input_value0 * sum1

        temp_product = tl.full(product2.shape, 0, product2.dtype)
        masked_product = tl.where(tmp2, product2, temp_product)
        broadcasted_product = tl.broadcast_to(masked_product, [XBLOCK, RBLOCK])
        temp_sum_update = temp_sum + broadcasted_product

        temp_sum = tl.where(r_mask & x_mask, temp_sum_update, temp_sum)

    final_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (x3), final_sum, x_mask)