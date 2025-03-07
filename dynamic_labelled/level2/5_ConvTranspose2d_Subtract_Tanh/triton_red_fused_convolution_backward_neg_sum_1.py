# From: 5_ConvTranspose2d_Subtract_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_neg_sum_1(in_ptr0, out_ptr0, out_ptr1, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 288
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 16
    input_index_0 = (input_index % 16)
    temp_accumulator_1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index
    temp_accumulator_2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index
        
        temp_index_0 = reduction_index_2 + input_index_1 * ((17 + kernel_size0 + 4 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size1 * kernel_size1) // 18)
        temp_index_1 = kernel_size0 + 4 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size1 * kernel_size1
        temp_mask_0 = temp_index_0 < temp_index_1
        
        temp_load = tl.load(
            in_ptr0 + (
                input_index_0 + 16 * (
                    (((temp_index_0 // (1 + 4 * kernel_size1 + 4 * kernel_size1 * kernel_size1)) % kernel_size0))
                ) + 2 * kernel_size1 * (
                    (((temp_index_0 // (1 + 2 * kernel_size1)) % (1 + 2 * kernel_size1)))
                ) + 4 * kernel_size1 * input_index_0 + 4 * input_index_0 * kernel_size1 * kernel_size1 + 64 * kernel_size1 * (
                    (((temp_index_0 // (1 + 4 * kernel_size1 + 4 * kernel_size1 * kernel_size1)) % kernel_size0))
                ) + 64 * kernel_size1 * kernel_size1 * (
                    (((temp_index_0 // (1 + 4 * kernel_size1 + 4 * kernel_size1 * kernel_size1)) % kernel_size0))
                ) + (((temp_index_0 % (1 + 2 * kernel_size1))) + ((((temp_index_0 // (1 + 2 * kernel_size1)) % (1 + 2 * kernel_size1)))))
            ),
            reduction_mask & temp_mask_0 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        
        temp_negate = -temp_load
        temp_zero = tl.full(temp_negate.shape, 0, temp_negate.dtype)
        temp_conditional = tl.where(temp_mask_0, temp_negate, temp_zero)
        temp_broadcast = tl.broadcast_to(temp_conditional, [XBLOCK, RBLOCK])
        temp_accumulate_1 = temp_accumulator_1 + temp_broadcast
        temp_accumulator_1 = tl.where(reduction_mask & input_mask, temp_accumulate_1, temp_accumulator_1)
        
        temp_broadcast_2 = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_accumulate_2 = temp_accumulator_2 + temp_broadcast_2
        temp_accumulator_2 = tl.where(reduction_mask & input_mask, temp_accumulate_2, temp_accumulator_2)
    
    temp_sum_1 = tl.sum(temp_accumulator_1, 1)[:, None]
    temp_sum_2 = tl.sum(temp_accumulator_2, 1)[:, None]
    
    tl.store(out_ptr0 + (input_index_3), temp_sum_1, input_mask)
    tl.store(out_ptr1 + (input_index_3), temp_sum_2, input_mask)