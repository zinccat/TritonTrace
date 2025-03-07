# From: 54_Conv2d_Multiply_LeakyReLU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_mul_sum_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 240
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 16
    input_index_0 = (input_index % 16)
    temp_buffer = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index

        temp_index_0 = reduction_index_2 + input_index_1 * (
            triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
        )
        temp_index_1 = 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1
        temp_mask_0 = temp_index_0 < temp_index_1

        temp_load_0 = tl.load(
            input_ptr0 + (
                (-2) * (((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // ((-2) + kernel_size1)) % ((-2) + kernel_size1))
                + 4 * input_index_0
                + 64 * (((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + kernel_size1 * (((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // ((-2) + kernel_size1)) % ((-2) + kernel_size1))
                + input_index_0 * kernel_size1 * kernel_size1
                + (-64) * kernel_size1 * (((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + (-4) * kernel_size1 * input_index_0
                + 16 * kernel_size1 * kernel_size1 * (((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + ((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) % ((-2) + kernel_size1))
            ),
            reduction_mask & temp_mask_0 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_load_1 = tl.load(
            input_ptr1 + (tl.broadcast_to(input_index_0, [XBLOCK, RBLOCK])),
            reduction_mask & temp_mask_0 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_product = temp_load_0 * temp_load_1

        temp_mask_1 = temp_product > 0.0

        temp_load_2 = tl.load(
            input_ptr2 + (
                (-2) * (((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // ((-2) + kernel_size1)) % ((-2) + kernel_size1))
                + 4 * input_index_0
                + 64 * (((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + kernel_size1 * (((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // ((-2) + kernel_size1)) % ((-2) + kernel_size1))
                + input_index_0 * kernel_size1 * kernel_size1
                + (-64) * kernel_size1 * (((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + (-4) * kernel_size1 * input_index_0
                + 16 * kernel_size1 * kernel_size1 * (((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + ((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) % ((-2) + kernel_size1))
            ),
            reduction_mask & temp_mask_0 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_constant_0 = 0.01
        temp_scaled_product = temp_product * temp_constant_0

        temp_selected_product = tl.where(temp_mask_1, temp_product, temp_scaled_product)

        temp_constant_1 = 0.7071067811865476
        temp_scaled_selected_product = temp_selected_product * temp_constant_1

        temp_erf = tl.extra.cuda.libdevice.erf(temp_scaled_selected_product)

        temp_constant_2 = 1.0
        temp_erf_plus_one = temp_erf + temp_constant_2

        temp_constant_3 = 0.5
        temp_half_erf_plus_one = temp_erf_plus_one * temp_constant_3

        temp_squared_selected_product = temp_selected_product * temp_selected_product

        temp_constant_4 = -0.5
        temp_exp_argument = temp_squared_selected_product * temp_constant_4

        temp_exp = tl.math.exp(temp_exp_argument)

        temp_constant_5 = 0.3989422804014327
        temp_scaled_exp = temp_exp * temp_constant_5

        temp_product_with_exp = temp_selected_product * temp_scaled_exp

        temp_gelu = temp_half_erf_plus_one + temp_product_with_exp

        temp_mask_2 = temp_mask_1

        temp_scaled_gelu = temp_gelu * temp_constant_0

        temp_selected_gelu = tl.where(temp_mask_1, temp_gelu, temp_scaled_gelu)

        temp_final_product = temp_selected_gelu * temp_load_0

        temp_broadcast_shape = temp_final_product.shape
        temp_broadcasted_product = tl.broadcast_to(temp_final_product, [XBLOCK, RBLOCK])

        temp_buffer += temp_broadcasted_product

        temp_buffer = tl.where(reduction_mask & input_mask, temp_buffer, temp_buffer)

    temp_sum = tl.sum(temp_buffer, 1)[:, None]
    tl.store(output_ptr0 + (input_index_3), temp_sum, input_mask)