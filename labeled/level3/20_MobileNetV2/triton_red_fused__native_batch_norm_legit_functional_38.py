# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_38red_fused__native_batch_norm_legit_functional_38(
    input_ptr, output_mean_ptr, output_var_ptr, output_count_ptr, 
    input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 1024
    reduction_num_elements = 123
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_block_row = input_index // 64
    input_block_col = (input_index % 64)
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    variance_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    count_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    full_input_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_flat = reduction_index
        flat_index = reduction_index_flat + 123 * input_block_row
        max_index = tl.full([1, 1], 1960, tl.int32)
        valid_index_mask = flat_index < max_index
        loaded_values = tl.load(
            input_ptr + (input_block_col + 64 * ((flat_index % 1960))),
            valid_index_mask & input_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        zero_values = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        valid_values = tl.where(valid_index_mask, loaded_values, zero_values)
        valid_mask = tl.full(valid_values.shape, 1, valid_values.dtype)
        invalid_mask = tl.where(valid_index_mask, valid_mask, zero_values)
        broadcasted_values = tl.broadcast_to(valid_values, [XBLOCK, RBLOCK])
        broadcasted_zero_values = tl.broadcast_to(zero_values, [XBLOCK, RBLOCK])
        broadcasted_valid_mask = tl.broadcast_to(invalid_mask, [XBLOCK, RBLOCK])
        
        mean_accumulator_next, variance_accumulator_next, count_accumulator_next = triton_helpers.welford_combine(
            mean_accumulator, variance_accumulator, count_accumulator,
            broadcasted_values, broadcasted_zero_values, broadcasted_valid_mask
        )
        
        mean_accumulator = tl.where(reduction_mask & input_mask, mean_accumulator_next, mean_accumulator)
        variance_accumulator = tl.where(reduction_mask & input_mask, variance_accumulator_next, variance_accumulator)
        count_accumulator = tl.where(reduction_mask & input_mask, count_accumulator_next, count_accumulator)

    mean_result, variance_result, count_result = triton_helpers.welford(
        mean_accumulator, variance_accumulator, count_accumulator, 1
    )
    
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    count_result = count_result[:, None]
    
    tl.store(output_mean_ptr + (full_input_index), mean_result, input_mask)
    tl.store(output_var_ptr + (full_input_index), variance_result, input_mask)
    tl.store(output_count_ptr + (full_input_index), count_result, input_mask)