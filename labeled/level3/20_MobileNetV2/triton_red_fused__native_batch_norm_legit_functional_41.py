# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_41(
    input_ptr, output_mean_ptr, output_var_ptr, output_count_ptr, 
    input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 6144
    reduction_num_elements = 123
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_channel = input_indices // 384
    input_position = (input_indices % 384)
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_linear_index = input_indices

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_index = reduction_indices
        temp_index = reduction_index + 123 * input_channel
        max_index = tl.full([1, 1], 1960, tl.int32)
        index_mask = temp_index < max_index
        temp_value = tl.load(
            input_ptr + (input_position + 384 * ((temp_index % 1960))),
            index_mask & reduction_mask & input_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        zero_value = tl.full(temp_value.shape, 0, temp_value.dtype)
        temp_zero = tl.where(index_mask, 0.0, zero_value)
        temp_one = tl.full(temp_value.shape, 0, temp_value.dtype)
        temp_one_mask = tl.where(index_mask, 1.0, temp_one)
        
        temp_value_broadcast = tl.broadcast_to(temp_value, [XBLOCK, RBLOCK])
        temp_zero_broadcast = tl.broadcast_to(temp_zero, [XBLOCK, RBLOCK])
        temp_one_mask_broadcast = tl.broadcast_to(temp_one_mask, [XBLOCK, RBLOCK])
        
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            temp_value_broadcast, temp_zero_broadcast, temp_one_mask_broadcast
        )
        
        temp_mean = tl.where(reduction_mask & input_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(reduction_mask & input_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(reduction_mask & input_mask, temp_weight_next, temp_weight)

    temp_mean_final, temp_m2_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    
    temp_mean_final_broadcast = temp_mean_final[:, None]
    temp_m2_final_broadcast = temp_m2_final[:, None]
    temp_weight_final_broadcast = temp_weight_final[:, None]
    
    tl.store(output_mean_ptr + (input_linear_index), temp_mean_final_broadcast, input_mask)
    tl.store(output_var_ptr + (input_linear_index), temp_m2_final_broadcast, input_mask)
    tl.store(output_count_ptr + (input_linear_index), temp_weight_final_broadcast, input_mask)