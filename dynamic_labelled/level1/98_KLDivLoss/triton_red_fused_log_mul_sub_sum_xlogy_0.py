# From: 98_KLDivLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_log_mul_sub_sum_xlogy_0red_fused_log_mul_sub_sum_xlogy_0(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
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
        temp_index = reduction_1 + input_0 * ((63 + kernel_size0 * kernel_size1) // 64)
        max_index = kernel_size0 * kernel_size1
        index_mask = temp_index < max_index
        
        loaded_value0 = tl.load(
            input_ptr0 + ((temp_index % max_index)), 
            reduction_mask & index_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        
        is_nan_mask = tl.extra.cuda.libdevice.isnan(loaded_value0).to(tl.int1)
        zero_value = 0.0
        is_zero_mask = loaded_value0 == zero_value
        log_value0 = tl.math.log(loaded_value0)
        product_log0 = loaded_value0 * log_value0
        safe_product_log0 = tl.where(is_zero_mask, zero_value, product_log0)
        nan_replacement = float("nan")
        safe_value0 = tl.where(is_nan_mask, nan_replacement, safe_product_log0)
        
        loaded_value1 = tl.load(
            input_ptr1 + ((temp_index % max_index)), 
            reduction_mask & index_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        
        log_value1 = tl.math.log(loaded_value1)
        product_log1 = loaded_value0 * log_value1
        result = safe_value0 - product_log1
        zero_result = tl.full(result.shape, 0, result.dtype)
        masked_result = tl.where(index_mask, result, zero_result)
        broadcast_result = tl.broadcast_to(masked_result, [XBLOCK, RBLOCK])
        
        temp_sum = temp_sum + broadcast_result
        temp_sum = tl.where(reduction_mask & input_mask, temp_sum, temp_sum)
    
    summed_result = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (input_0), summed_result, input_mask)