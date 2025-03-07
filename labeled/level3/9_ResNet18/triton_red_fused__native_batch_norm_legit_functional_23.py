# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_23(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    num_elements, num_reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 1024
    num_reduction_elements = 98
    element_offset = tl.program_id(0) * XBLOCK
    element_indices = element_offset + tl.arange(0, XBLOCK)[:, None]
    element_mask = element_indices < num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    element_index_mod = element_indices % 256
    element_index_div = element_indices // 256
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    element_index_flat = element_indices

    for reduction_offset in range(0, num_reduction_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < num_reduction_elements
        reduction_index_flat = reduction_indices
        input_data = tl.load(
            input_ptr + (element_index_mod + 256 * reduction_index_flat + 25088 * element_index_div), 
            reduction_mask & element_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcasted_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_input, mean_accumulator, m2_accumulator, weight_accumulator, reduction_offset == 0
        )
        mean_accumulator = tl.where(reduction_mask & element_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(reduction_mask & element_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(reduction_mask & element_mask, weight_next, weight_accumulator)

    mean_result, variance_result, weight_result = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    weight_result = weight_result[:, None]

    tl.store(output_mean_ptr + (element_index_flat), mean_result, element_mask)
    tl.store(output_var_ptr + (element_index_flat), variance_result, element_mask)
    tl.store(output_weight_ptr + (element_index_flat), weight_result, element_mask)