# From: 95_CrossEntropyLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_0red_fused__log_softmax_0(
    input_ptr, output_ptr_log, output_ptr_exp_sum, kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices_0 = input_indices
    max_values = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_indices_1 = reduction_indices
        loaded_values = tl.load(input_ptr + (reduction_indices_1 + kernel_size * input_indices_0), reduction_mask & input_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        max_values = triton_helpers.maximum(max_values, broadcasted_values)
        max_values = tl.where(reduction_mask & input_mask, max_values, max_values)
    
    max_values_per_input = triton_helpers.max2(max_values, 1)[:, None]
    tl.store(output_ptr_log + (input_indices_0), max_values_per_input, input_mask)
    
    exp_sum_values = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_indices_1 = reduction_indices
        loaded_values = tl.load(input_ptr + (reduction_indices_1 + kernel_size * input_indices_0), reduction_mask & input_mask, eviction_policy='evict_first', other=0.0)
        adjusted_values = loaded_values - max_values_per_input
        exp_values = tl.math.exp(adjusted_values)
        broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
        exp_sum_values = exp_sum_values + broadcasted_exp_values
        exp_sum_values = tl.where(reduction_mask & input_mask, exp_sum_values, exp_sum_values)
    
    exp_sum_per_input = tl.sum(exp_sum_values, 1)[:, None]
    tl.store(output_ptr_exp_sum + (input_indices_0), exp_sum_per_input, input_mask)