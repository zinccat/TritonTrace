# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_64red_fused__native_batch_norm_legit_functional_64(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 6400
    reduction_elements = 128
    element_offset = tl.program_id(0) * XBLOCK
    element_indices = element_offset + tl.arange(0, XBLOCK)[:, None]
    element_mask = element_indices < total_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    element_index_0 = (element_indices % 1280)
    element_index_1 = element_indices // 1280
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    element_index_3 = element_indices

    for reduction_offset in range(0, reduction_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_elements
        reduction_index_2 = reduction_indices
        loaded_values = tl.load(
            input_ptr + (element_index_0 + 1280 * reduction_index_2 + 163840 * element_index_1), 
            reduction_mask & element_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcasted_values, running_mean, running_m2, running_weight, reduction_offset == 0
        )
        running_mean = tl.where(reduction_mask & element_mask, running_mean_next, running_mean)
        running_m2 = tl.where(reduction_mask & element_mask, running_m2_next, running_m2)
        running_weight = tl.where(reduction_mask & element_mask, running_weight_next, running_weight)

    final_mean, final_var, final_weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )
    final_mean = final_mean[:, None]
    final_var = final_var[:, None]
    final_weight = final_weight[:, None]

    tl.store(output_mean_ptr + (element_index_3), final_mean, element_mask)
    tl.store(output_var_ptr + (element_index_3), final_var, element_mask)
    tl.store(output_weight_ptr + (element_index_3), final_weight, element_mask)