# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_div_native_batch_norm_backward_1(
    input_grad_ptr, input_data_ptr, input_mean_ptr, input_var_ptr, 
    output_grad_ptr0, output_grad_ptr1, kernel_size, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 352
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_32 = x_index // 32
    x_mod_32 = x_index % 32
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_3d_index = x_index
    temp_accumulator2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_2d_index = r_index
        temp_index = (
            r_2d_index + 46 * x_div_32 + 
            x_div_32 * triton_helpers.div_floor_integer(
                ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size, 11
            )
        )
        temp_limit = 496 + ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size
        temp_condition = temp_index < temp_limit

        input_grad = tl.load(
            input_grad_ptr + (
                (-1) * (((temp_index // ((-1) + 2 * kernel_size)) % ((-1) + 2 * kernel_size))) 
                + 31 * x_mod_32 
                + 992 * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16)) 
                + ((-3968) * kernel_size * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16))) 
                + ((-124) * kernel_size * x_mod_32) 
                + ((-4) * kernel_size * (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31))) 
                + 2 * kernel_size * (((temp_index // ((-1) + 2 * kernel_size)) % ((-1) + 2 * kernel_size))) 
                + 4 * kernel_size * kernel_size * (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31)) 
                + 124 * x_mod_32 * kernel_size * kernel_size 
                + 3968 * kernel_size * kernel_size * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16)) 
                + ((temp_index % ((-1) + 2 * kernel_size))) 
                + (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31))
            ),
            r_mask & temp_condition & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        input_data = tl.load(
            input_data_ptr + (x_mod_32 + 32 * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16))),
            r_mask & temp_condition & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )

        divisor = tl.broadcast_to(31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size, [XBLOCK, RBLOCK]).to(tl.float32)
        normalized_input = input_data / divisor
        temp_sum = input_grad + normalized_input
        temp_broadcast = tl.full(temp_sum.shape, 0, temp_sum.dtype)
        temp_result = tl.where(temp_condition, temp_sum, temp_broadcast)
        temp_broadcasted = tl.broadcast_to(temp_result, [XBLOCK, RBLOCK])
        temp_accumulator += temp_broadcasted
        temp_accumulator = tl.where(r_mask & x_mask, temp_accumulator, temp_accumulator)

        input_mean = tl.load(
            input_mean_ptr + (
                (-1) * (((temp_index // ((-1) + 2 * kernel_size)) % ((-1) + 2 * kernel_size))) 
                + 31 * x_mod_32 
                + 992 * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16)) 
                + ((-3968) * kernel_size * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16))) 
                + ((-124) * kernel_size * x_mod_32) 
                + ((-4) * kernel_size * (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31))) 
                + 2 * kernel_size * (((temp_index // ((-1) + 2 * kernel_size)) % ((-1) + 2 * kernel_size))) 
                + 4 * kernel_size * kernel_size * (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31)) 
                + 124 * x_mod_32 * kernel_size * kernel_size 
                + 3968 * kernel_size * kernel_size * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16)) 
                + ((temp_index % ((-1) + 2 * kernel_size))) 
                + (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31))
            ),
            r_mask & temp_condition & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        input_var = tl.load(
            input_var_ptr + (tl.broadcast_to(x_mod_32, [XBLOCK, RBLOCK])),
            r_mask & temp_condition & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_diff = temp_sum - input_var
        temp_product = temp_sum * temp_diff
        temp_product_broadcast = tl.full(temp_product.shape, 0, temp_product.dtype)
        temp_product_result = tl.where(temp_condition, temp_product, temp_product_broadcast)
        temp_product_broadcasted = tl.broadcast_to(temp_product_result, [XBLOCK, RBLOCK])
        temp_accumulator2 += temp_product_broadcasted
        temp_accumulator2 = tl.where(r_mask & x_mask, temp_accumulator2, temp_accumulator2)

    final_sum0 = tl.sum(temp_accumulator, 1)[:, None]
    final_sum1 = tl.sum(temp_accumulator2, 1)[:, None]
    tl.store(output_grad_ptr0 + (x_3d_index), final_sum0, x_mask)
    tl.store(output_grad_ptr1 + (x_3d_index), final_sum1, x_mask)