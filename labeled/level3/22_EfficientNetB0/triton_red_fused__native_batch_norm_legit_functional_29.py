# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_29(
    input_ptr, output_mean_ptr, output_var_ptr, output_count_ptr, 
    input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 8928
    reduction_num_elements = 127
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_channel = input_indices // 144
    input_position = input_indices % 144
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_linear_index = input_indices

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_index = reduction_indices
        temp_index = reduction_index + 127 * input_channel
        max_index = tl.full([1, 1], 7840, tl.int32)
        index_mask = temp_index < max_index
        temp_data = tl.load(
            input_ptr + (input_position + 144 * ((temp_index % 7840))),
            reduction_mask & index_mask & input_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        temp_zero = tl.where(index_mask, zero_value, zero_broadcast)
        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        temp_one = tl.where(index_mask, one_value, one_broadcast)
        temp_data_broadcast = tl.broadcast_to(temp_data, [XBLOCK, RBLOCK])
        temp_zero_broadcast = tl.broadcast_to(temp_zero, [XBLOCK, RBLOCK])
        temp_one_broadcast = tl.broadcast_to(temp_one, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            temp_data_broadcast, temp_zero_broadcast, temp_one_broadcast
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