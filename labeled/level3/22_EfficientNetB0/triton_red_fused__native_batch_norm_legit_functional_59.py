# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_59red_fused__native_batch_norm_legit_functional_59(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 768
    reduction_num_elements = 123
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_channel = input_indices // 192
    input_position = input_indices % 192
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_linear_index = input_indices

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_index = reduction_indices
        temp_index = reduction_index + 123 * input_channel
        max_index = tl.full([1, 1], 490, tl.int32)
        valid_mask = temp_index < max_index
        loaded_values = tl.load(
            input_ptr + (input_position + 192 * ((temp_index % 490))),
            valid_mask & reduction_mask & input_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        zero_values = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        valid_values = tl.where(valid_mask, 0.0, zero_values)
        valid_weights = tl.where(valid_mask, 1.0, zero_values)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcasted_values_zero = tl.broadcast_to(valid_values, [XBLOCK, RBLOCK])
        broadcasted_weights = tl.broadcast_to(valid_weights, [XBLOCK, RBLOCK])
        
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            broadcasted_values, broadcasted_values_zero, broadcasted_weights
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
    tl.store(output_weight_ptr + (input_linear_index), final_weight, input_mask)