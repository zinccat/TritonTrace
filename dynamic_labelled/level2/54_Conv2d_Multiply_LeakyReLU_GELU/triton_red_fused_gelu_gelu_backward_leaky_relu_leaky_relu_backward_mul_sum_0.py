# From: 54_Conv2d_Multiply_LeakyReLU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_mul_sum_0red_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_mul_sum_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    input_num_elements = 240
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_x1 = input_index // 16
    input_x0 = input_index % 16
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_x3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_r2 = reduction_index

        temp_index = reduction_r2 + input_x1 * (triton_helpers.div_floor_integer(
            14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15))

        temp_limit = 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1
        temp_condition = temp_index < temp_limit

        temp_load0 = tl.load(input_ptr0 + (((-2) * (((temp_index // ((-2) + kernel_size1)) % ((-2) + kernel_size1)))) + 
                                           4 * input_x0 + 64 * (((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)) + 
                                           kernel_size1 * (((temp_index // ((-2) + kernel_size1)) % ((-2) + kernel_size1))) + 
                                           input_x0 * kernel_size1 * kernel_size1 + 
                                           (-64) * kernel_size1 * (((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)) + 
                                           (-4) * kernel_size1 * input_x0 + 
                                           16 * kernel_size1 * kernel_size1 * (((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)) + 
                                           (temp_index % ((-2) + kernel_size1))), 
                            reduction_mask & temp_condition & input_mask, 
                            eviction_policy='evict_last', other=0.0)

        temp_load1 = tl.load(input_ptr1 + (tl.broadcast_to(input_x0, [XBLOCK, RBLOCK])), 
                             reduction_mask & temp_condition & input_mask, 
                             eviction_policy='evict_last', other=0.0)

        temp_product = temp_load0 * temp_load1

        temp_zero = 0.0
        temp_greater = temp_product > temp_zero

        temp_load2 = tl.load(input_ptr2 + (((-2) * (((temp_index // ((-2) + kernel_size1)) % ((-2) + kernel_size1)))) + 
                                           4 * input_x0 + 64 * (((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)) + 
                                           kernel_size1 * (((temp_index // ((-2) + kernel_size1)) % ((-2) + kernel_size1))) + 
                                           input_x0 * kernel_size1 * kernel_size1 + 
                                           (-64) * kernel_size1 * (((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)) + 
                                           (-4) * kernel_size1 * input_x0 + 
                                           16 * kernel_size1 * kernel_size1 * (((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)) + 
                                           (temp_index % ((-2) + kernel_size1))), 
                            reduction_mask & temp_condition & input_mask, 
                            eviction_policy='evict_last', other=0.0)

        temp_alpha = 0.01
        temp_scaled_product = temp_product * temp_alpha

        temp_select = tl.where(temp_greater, temp_product, temp_scaled_product)

        temp_beta = 0.7071067811865476
        temp_scaled_select = temp_select * temp_beta

        temp_erf = tl.extra.cuda.libdevice.erf(temp_scaled_select)

        temp_one = 1.0
        temp_erf_plus_one = temp_erf + temp_one

        temp_half = 0.5
        temp_half_erf_plus_one = temp_erf_plus_one * temp_half

        temp_squared_select = temp_select * temp_select

        temp_neg_half = -0.5
        temp_exp_component = temp_squared_select * temp_neg_half

        temp_exp = tl.math.exp(temp_exp_component)

        temp_coefficient = 0.3989422804014327
        temp_gaussian = temp_exp * temp_coefficient

        temp_gaussian_component = temp_select * temp_gaussian

        temp_final = temp_half_erf_plus_one + temp_gaussian_component

        temp_conditioned_final = temp_greater * temp_final

        temp_conditioned_scaled_final = temp_conditioned_final * temp_alpha

        temp_final_select = tl.where(temp_greater, temp_conditioned_final, temp_conditioned_scaled_final)

        temp_final_product = temp_final_select * temp_load0

        temp_broadcast_final_product = tl.broadcast_to(temp_final_product, [XBLOCK, RBLOCK])

        temp_accumulator += temp_broadcast_final_product

        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulator, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (input_x3), temp_sum, input_mask)