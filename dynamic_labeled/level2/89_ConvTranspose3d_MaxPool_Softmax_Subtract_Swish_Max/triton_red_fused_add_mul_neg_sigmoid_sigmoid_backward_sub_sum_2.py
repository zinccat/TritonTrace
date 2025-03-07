# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_mul_neg_sigmoid_sigmoid_backward_sub_sum_2(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 16
    input_index_0 = (input_index % 16)
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index
        temp_index0 = reduction_index_2 + input_index_1 * ((20 + kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2) // 21)
        temp_index1 = kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2
        temp_mask = temp_index0 < temp_index1
        temp_load0 = tl.load(
            input_ptr0 + (kernel_size1 * input_index_0 * kernel_size2 * kernel_size2 + 
                          16 * kernel_size1 * kernel_size2 * kernel_size2 * 
                          (((temp_index0 // kernel_size3) % kernel_size0) + 
                           (temp_index0 % kernel_size3))),
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        temp_load1 = tl.load(
            input_ptr1 + (kernel_size1 * input_index_0 * kernel_size2 * kernel_size2 + 
                          16 * kernel_size1 * kernel_size2 * kernel_size2 * 
                          (((temp_index0 // kernel_size3) % kernel_size0) + 
                           (temp_index0 % kernel_size3))),
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        temp_load2 = tl.load(
            input_ptr2 + (tl.broadcast_to(input_index_0, [XBLOCK, RBLOCK])),
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        temp_diff = temp_load1 - temp_load2
        temp_sigmoid = tl.sigmoid(temp_diff)
        temp_product1 = temp_load0 * temp_sigmoid
        temp_product2 = temp_load0 * temp_diff
        temp_one = 1.0
        temp_complement = temp_one - temp_sigmoid
        temp_product3 = temp_sigmoid * temp_complement
        temp_product4 = temp_product2 * temp_product3
        temp_sum = temp_product1 + temp_product4
        temp_negate = -temp_sum
        temp_zero = tl.full(temp_negate.shape, 0, temp_negate.dtype)
        temp_conditional = tl.where(temp_mask, temp_negate, temp_zero)
        temp_broadcast = tl.broadcast_to(temp_conditional, [XBLOCK, RBLOCK])
        temp_accumulate = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulate, temp_accumulator)

    temp_sum_reduction = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (input_index_3), temp_sum_reduction, input_mask)