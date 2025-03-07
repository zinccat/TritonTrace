# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_18(
    input_ptr, output_mean_ptr, output_var_ptr, output_count_ptr, 
    input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 1664
    reduction_num_elements = 121
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_channel = input_indices // 128
    input_within_channel = input_indices % 128
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_linear_index = input_indices

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_linear_index = reduction_indices
        temp_index = reduction_linear_index + 121 * input_channel
        max_index = tl.full([1, 1], 1568, tl.int32)
        index_mask = temp_index < max_index
        loaded_values = tl.load(
            input_ptr + (input_within_channel + 128 * ((temp_index % 1568))),
            index_mask & reduction_mask & input_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        zero_values = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        ones_values = tl.full(zero_values.shape, 1, zero_values.dtype)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcasted_zeros = tl.broadcast_to(zero_values, [XBLOCK, RBLOCK])
        broadcasted_ones = tl.broadcast_to(ones_values, [XBLOCK, RBLOCK])
        
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            broadcasted_values, broadcasted_zeros, broadcasted_ones
        )
        
        temp_mean = tl.where(reduction_mask & input_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(reduction_mask & input_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(reduction_mask & input_mask, temp_weight_next, temp_weight)

    final_mean, final_m2, final_weight = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    
    final_mean = final_mean[:, None]
    final_m2 = final_m2[:, None]
    final_weight = final_weight[:, None]
    
    tl.store(output_mean_ptr + (input_linear_index), final_mean, input_mask)
    tl.store(output_var_ptr + (input_linear_index), final_m2, input_mask)
    tl.store(output_count_ptr + (input_linear_index), final_weight, input_mask)