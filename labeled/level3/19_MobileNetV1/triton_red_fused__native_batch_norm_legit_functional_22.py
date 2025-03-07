# From: 19_MobileNetV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_22(
    input_ptr, output_mean_ptr, output_var_ptr, output_count_ptr, 
    input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 15872
    reduction_num_elements = 127
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_channel = input_indices // 256
    input_within_channel = (input_indices % 256)
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_linear_index = input_indices

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_linear_index = reduction_indices
        temp_index = reduction_linear_index + 127 * input_channel
        max_index = tl.full([1, 1], 7840, tl.int32)
        index_within_max = temp_index < max_index
        temp_load = tl.load(
            input_ptr + (input_within_channel + 256 * ((temp_index % 7840))),
            index_within_max & input_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        temp_zero = tl.full(temp_load.shape, 0, temp_load.dtype)
        temp_select_zero = tl.where(index_within_max, 0.0, temp_zero)
        temp_select_one = tl.where(index_within_max, 1.0, temp_zero)
        temp_broadcast_load = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_broadcast_zero = tl.broadcast_to(temp_select_zero, [XBLOCK, RBLOCK])
        temp_broadcast_one = tl.broadcast_to(temp_select_one, [XBLOCK, RBLOCK])
        
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            temp_broadcast_load, temp_broadcast_zero, temp_broadcast_one
        )
        
        temp_mean = tl.where(reduction_mask & input_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(reduction_mask & input_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(reduction_mask & input_mask, temp_weight_next, temp_weight)

    temp_mean_final, temp_m2_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    
    temp_mean_final = temp_mean_final[:, None]
    temp_m2_final = temp_m2_final[:, None]
    temp_weight_final = temp_weight_final[:, None]
    
    tl.store(output_mean_ptr + (input_linear_index), temp_mean_final, input_mask)
    tl.store(output_var_ptr + (input_linear_index), temp_m2_final, input_mask)
    tl.store(output_count_ptr + (input_linear_index), temp_weight_final, input_mask)