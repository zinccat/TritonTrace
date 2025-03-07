# From: 37_FrobeniusNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_linalg_vector_norm_0red_fused_linalg_vector_norm_0(
    input_ptr, output_ptr, kernel_dim0, kernel_dim1, kernel_dim2, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 328
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_0 = input_index
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_1 = reduction_index
        temp_index = reduction_1 + input_0 * ((327 + kernel_dim0 * kernel_dim1 * kernel_dim2 * kernel_dim2) // 328)
        total_elements = kernel_dim0 * kernel_dim1 * kernel_dim2 * kernel_dim2
        within_bounds = temp_index < total_elements
        loaded_values = tl.load(
            input_ptr + ((temp_index % total_elements)), 
            reduction_mask & within_bounds & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        squared_values = loaded_values * loaded_values
        zero_filled = tl.full(squared_values.shape, 0, squared_values.dtype)
        masked_squared_values = tl.where(within_bounds, squared_values, zero_filled)
        broadcasted_values = tl.broadcast_to(masked_squared_values, [XBLOCK, RBLOCK])
        temp_sum += broadcasted_values
        temp_sum = tl.where(reduction_mask & input_mask, temp_sum, temp_sum)
    
    result_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr + (input_0), result_sum, input_mask)