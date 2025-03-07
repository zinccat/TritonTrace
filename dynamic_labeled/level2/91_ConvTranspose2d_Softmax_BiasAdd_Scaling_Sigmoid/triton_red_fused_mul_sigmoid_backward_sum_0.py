# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sigmoid_backward_sum_0(
    input_grad_ptr, input_data_ptr, output_grad_ptr, kernel_size_0, kernel_size_1, 
    input_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_elements = 448
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_block_1 = input_index // 64
    input_block_0 = (input_index % 64)
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_block_3 = input_index

    for reduction_offset in range(0, reduction_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_elements
        reduction_block_2 = reduction_index

        temp_index_0 = reduction_block_2 + input_block_1 * ((6 + kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1) // 7)
        temp_index_1 = kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1
        temp_mask_2 = temp_index_0 < temp_index_1

        input_grad_value = tl.load(
            input_grad_ptr + (input_block_0 + 64 * (((temp_index_0 // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
            2 * kernel_size_1 * (((temp_index_0 // (1 + 2 * kernel_size_1)) % (1 + 2 * kernel_size_1))) + 
            4 * kernel_size_1 * input_block_0 + 4 * input_block_0 * kernel_size_1 * kernel_size_1 + 
            256 * kernel_size_1 * ((temp_index_0 // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0) + 
            256 * kernel_size_1 * kernel_size_1 * ((temp_index_0 // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0) + 
            (temp_index_0 % (1 + 2 * kernel_size_1)) + 
            (((temp_index_0 // (1 + 2 * kernel_size_1)) % (1 + 2 * kernel_size_1)))), 
            reduction_mask & temp_mask_2 & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        input_data_value = tl.load(
            input_data_ptr + (input_block_0 + 64 * (((temp_index_0 // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
            2 * kernel_size_1 * (((temp_index_0 // (1 + 2 * kernel_size_1)) % (1 + 2 * kernel_size_1))) + 
            4 * kernel_size_1 * input_block_0 + 4 * input_block_0 * kernel_size_1 * kernel_size_1 + 
            256 * kernel_size_1 * ((temp_index_0 // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0) + 
            256 * kernel_size_1 * kernel_size_1 * ((temp_index_0 // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0) + 
            (temp_index_0 % (1 + 2 * kernel_size_1)) + 
            (((temp_index_0 // (1 + 2 * kernel_size_1)) % (1 + 2 * kernel_size_1)))), 
            reduction_mask & temp_mask_2 & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        temp_value_5 = 1.0
        temp_value_6 = temp_value_5 - input_data_value
        temp_value_7 = input_data_value * temp_value_6
        temp_value_8 = input_grad_value * temp_value_7
        temp_value_9 = 2.0
        temp_value_10 = temp_value_8 * temp_value_9
        temp_value_11 = tl.full(temp_value_10.shape, 0, temp_value_10.dtype)
        temp_value_12 = tl.where(temp_mask_2, temp_value_10, temp_value_11)
        temp_value_13 = tl.broadcast_to(temp_value_12, [XBLOCK, RBLOCK])
        temp_value_15 = temp_accumulator + temp_value_13
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_value_15, temp_accumulator)

    temp_value_14 = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_grad_ptr + (input_block_3), temp_value_14, input_mask)