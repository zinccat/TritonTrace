# From: 94_MSELoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_pow_sub_0red_fused_mean_pow_sub_0(input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 64
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_0 = input_index
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_1 = reduction_index
        temp_index = reduction_1 + input_0 * ((63 + kernel_size0 * kernel_size1) // 64)
        temp_kernel_size = kernel_size0 * kernel_size1
        temp_condition = temp_index < temp_kernel_size
        temp_value0 = tl.load(input_ptr0 + ((temp_index % temp_kernel_size)), reduction_mask & temp_condition & input_mask, eviction_policy='evict_last', other=0.0)
        temp_value1 = tl.load(input_ptr1 + ((temp_index % temp_kernel_size)), reduction_mask & temp_condition & input_mask, eviction_policy='evict_last', other=0.0)
        temp_difference = temp_value0 - temp_value1
        temp_squared = temp_difference * temp_difference
        temp_zero_filled = tl.full(temp_squared.shape, 0, temp_squared.dtype)
        temp_selected = tl.where(temp_condition, temp_squared, temp_zero_filled)
        temp_broadcasted = tl.broadcast_to(temp_selected, [XBLOCK, RBLOCK])
        temp_accumulated = temp_accumulator + temp_broadcasted
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulated, temp_accumulator)
    
    temp_summed = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (input_0), temp_summed, input_mask)