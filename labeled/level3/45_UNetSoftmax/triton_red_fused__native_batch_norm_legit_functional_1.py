# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_mean_ptr, output_variance_ptr, output_count_ptr, 
    num_elements, num_reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 384
    num_reduction_elements = 43691
    element_offset = tl.program_id(0) * XBLOCK
    element_indices = element_offset + tl.arange(0, XBLOCK)[:, None]
    element_mask = element_indices < num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    element_index_1 = element_indices // 64
    element_index_0 = (element_indices % 64)
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    element_index_3 = element_indices

    for reduction_offset in range(0, num_reduction_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < num_reduction_elements
        reduction_index_2 = reduction_indices
        combined_index = reduction_index_2 + 43691 * element_index_1
        max_index_limit = tl.full([1, 1], 262144, tl.int32)
        index_within_limit = combined_index < max_index_limit
        loaded_values = tl.load(
            input_ptr + (32768 * element_index_0 + 2097152 * (((combined_index // 32768) % 8)) + (combined_index % 32768)),
            reduction_mask & index_within_limit & element_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        zero_values = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        mask_values = tl.where(index_within_limit, 0.0, zero_values)
        one_values = tl.full(loaded_values.shape, 0, 1.0)
        mask_ones = tl.where(index_within_limit, 1.0, one_values)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcasted_mask_values = tl.broadcast_to(mask_values, [XBLOCK, RBLOCK])
        broadcasted_mask_ones = tl.broadcast_to(mask_ones, [XBLOCK, RBLOCK])
        
        mean_next, m2_next, weight_next = triton_helpers.welford_combine(
            mean_accumulator, m2_accumulator, weight_accumulator,
            broadcasted_values, broadcasted_mask_values, broadcasted_mask_ones
        )
        
        mean_accumulator = tl.where(reduction_mask & element_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(reduction_mask & element_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(reduction_mask & element_mask, weight_next, weight_accumulator)

    mean_result, variance_result, count_result = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    count_result = count_result[:, None]
    
    tl.store(output_mean_ptr + (element_index_3), mean_result, element_mask)
    tl.store(output_variance_ptr + (element_index_3), variance_result, element_mask)
    tl.store(output_count_ptr + (element_index_3), count_result, element_mask)