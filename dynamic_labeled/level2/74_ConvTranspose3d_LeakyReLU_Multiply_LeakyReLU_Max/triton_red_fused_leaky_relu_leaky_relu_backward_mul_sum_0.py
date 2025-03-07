# From: 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_leaky_relu_leaky_relu_backward_mul_sum_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 352
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 32
    input_index_0 = (input_index % 32)
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index
        temp_index_0 = reduction_index_2 + input_index_1 * ((10 + 2048 * kernel_size * kernel_size) // 11)
        temp_index_1 = 2048 * kernel_size * kernel_size
        temp_mask = temp_index_0 < temp_index_1
        temp_value_0 = tl.load(
            input_ptr0 + (
                128 * input_index_0 * kernel_size * kernel_size + 
                4096 * kernel_size * kernel_size * (
                    ((reduction_index_2 + input_index_1 * ((10 + 2048 * kernel_size * kernel_size) // 11)) // (128 * kernel_size * kernel_size)) % 16
                ) + 
                ((reduction_index_2 + input_index_1 * ((10 + 2048 * kernel_size * kernel_size) // 11)) % (128 * kernel_size * kernel_size))
            ), 
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        temp_value_1 = 0.0
        temp_mask_1 = temp_value_0 > temp_value_1
        leaky_relu_slope = 0.2
        temp_value_2 = temp_value_0 * leaky_relu_slope
        temp_value_3 = tl.where(temp_mask_1, temp_value_0, temp_value_2)
        temp_value_4 = tl.load(
            input_ptr1 + (tl.broadcast_to(input_index_0, [XBLOCK, RBLOCK])), 
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        temp_value_5 = temp_value_3 * temp_value_4
        temp_mask_2 = temp_value_5 > temp_value_1
        temp_value_6 = tl.load(
            input_ptr2 + (
                128 * input_index_0 * kernel_size * kernel_size + 
                4096 * kernel_size * kernel_size * (
                    ((reduction_index_2 + input_index_1 * ((10 + 2048 * kernel_size * kernel_size) // 11)) // (128 * kernel_size * kernel_size)) % 16
                ) + 
                ((reduction_index_2 + input_index_1 * ((10 + 2048 * kernel_size * kernel_size) // 11)) % (128 * kernel_size * kernel_size))
            ), 
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        temp_value_7 = temp_mask_2 * leaky_relu_slope
        temp_value_8 = tl.where(temp_mask_2, temp_value_6, temp_value_7)
        temp_value_9 = temp_value_8 * temp_value_3
        temp_value_10 = tl.full(temp_value_9.shape, 0, temp_value_9.dtype)
        temp_value_11 = tl.where(temp_mask, temp_value_9, temp_value_10)
        temp_value_12 = tl.broadcast_to(temp_value_11, [XBLOCK, RBLOCK])
        temp_value_13 = temp_accumulator + temp_value_12
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_value_13, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (input_index_3), temp_result, input_mask)