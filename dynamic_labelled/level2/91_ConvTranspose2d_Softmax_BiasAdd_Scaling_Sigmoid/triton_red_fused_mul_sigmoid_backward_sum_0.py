# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sigmoid_backward_sum_0red_fused_mul_sigmoid_backward_sum_0(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    input_num_elements = 448
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 64
    input_index_0 = (input_index % 64)
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index

        temp_index0 = reduction_index_2 + input_index_1 * ((6 + kernel_size0 + 4 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size1 * kernel_size1) // 7)
        temp_index1 = kernel_size0 + 4 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size1 * kernel_size1
        temp_mask = temp_index0 < temp_index1

        input_value0 = tl.load(
            input_ptr0 + (input_index_0 + 64 * (((temp_index0 // (1 + 4 * kernel_size1 + 4 * kernel_size1 * kernel_size1)) % kernel_size0)) 
            + 2 * kernel_size1 * (((temp_index0 // (1 + 2 * kernel_size1)) % (1 + 2 * kernel_size1))) 
            + 4 * kernel_size1 * input_index_0 + 4 * input_index_0 * kernel_size1 * kernel_size1 
            + 256 * kernel_size1 * ((temp_index0 // (1 + 4 * kernel_size1 + 4 * kernel_size1 * kernel_size1)) % kernel_size0) 
            + 256 * kernel_size1 * kernel_size1 * ((temp_index0 // (1 + 4 * kernel_size1 + 4 * kernel_size1 * kernel_size1)) % kernel_size0) 
            + (temp_index0 % (1 + 2 * kernel_size1)) 
            + (((temp_index0 // (1 + 2 * kernel_size1)) % (1 + 2 * kernel_size1)))), 
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        input_value1 = tl.load(
            input_ptr1 + (input_index_0 + 64 * (((temp_index0 // (1 + 4 * kernel_size1 + 4 * kernel_size1 * kernel_size1)) % kernel_size0)) 
            + 2 * kernel_size1 * (((temp_index0 // (1 + 2 * kernel_size1)) % (1 + 2 * kernel_size1))) 
            + 4 * kernel_size1 * input_index_0 + 4 * input_index_0 * kernel_size1 * kernel_size1 
            + 256 * kernel_size1 * ((temp_index0 // (1 + 4 * kernel_size1 + 4 * kernel_size1 * kernel_size1)) % kernel_size0) 
            + 256 * kernel_size1 * kernel_size1 * ((temp_index0 // (1 + 4 * kernel_size1 + 4 * kernel_size1 * kernel_size1)) % kernel_size0) 
            + (temp_index0 % (1 + 2 * kernel_size1)) 
            + (((temp_index0 // (1 + 2 * kernel_size1)) % (1 + 2 * kernel_size1)))), 
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        temp_value1 = 1.0
        temp_value2 = temp_value1 - input_value1
        temp_value3 = input_value1 * temp_value2
        temp_value4 = input_value0 * temp_value3
        temp_value5 = 2.0
        temp_value6 = temp_value4 * temp_value5
        temp_value7 = tl.full(temp_value6.shape, 0, temp_value6.dtype)
        temp_value8 = tl.where(temp_mask, temp_value6, temp_value7)
        temp_value9 = tl.broadcast_to(temp_value8, [XBLOCK, RBLOCK])
        temp_value10 = temp_accumulator + temp_value9
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_value10, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (input_index_3), temp_result, input_mask)