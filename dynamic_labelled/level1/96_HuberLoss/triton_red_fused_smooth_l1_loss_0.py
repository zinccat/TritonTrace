# From: 96_HuberLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_smooth_l1_loss_0red_fused_smooth_l1_loss_0(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, 
    RBLOCK: tl.constexpr
):
    input_num_elements = 64
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_0 = input_index
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_1 = reduction_index
        combined_index = reduction_1 + input_0 * ((63 + kernel_size0 * kernel_size1) // 64)
        kernel_product = kernel_size0 * kernel_size1
        index_within_bounds = combined_index < kernel_product
        
        input_value0 = tl.load(
            input_ptr0 + ((combined_index % kernel_product)), 
            reduction_mask & index_within_bounds & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        input_value1 = tl.load(
            input_ptr1 + ((combined_index % kernel_product)), 
            reduction_mask & index_within_bounds & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        
        difference = input_value0 - input_value1
        absolute_difference = tl.math.abs(difference)
        threshold = 1.0
        is_within_threshold = absolute_difference < threshold
        squared_difference = absolute_difference * absolute_difference
        half = 0.5
        smooth_l1_loss = squared_difference * half * threshold
        linear_loss = absolute_difference - half
        
        smooth_l1_result = tl.where(is_within_threshold, smooth_l1_loss, linear_loss)
        zero_filled = tl.full(smooth_l1_result.shape, 0, smooth_l1_result.dtype)
        masked_result = tl.where(index_within_bounds, smooth_l1_result, zero_filled)
        broadcasted_result = tl.broadcast_to(masked_result, [XBLOCK, RBLOCK])
        
        temp_sum += broadcasted_result
        temp_sum = tl.where(reduction_mask & input_mask, temp_sum, temp_sum)
    
    reduced_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (input_0), reduced_sum, input_mask)