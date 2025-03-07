# From: 39_L2Norm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_linalg_vector_norm_0(in_ptr0, out_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_mod2 = input_indices % 2
    input_index_div2 = input_indices // 2
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index = input_indices

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_index = reduction_indices
        temp_index = reduction_index + input_index_mod2 * ((1 + kernel_size) // 2)
        kernel_size_temp = kernel_size
        index_within_bounds = temp_index < kernel_size_temp
        loaded_value = tl.load(in_ptr0 + (reduction_index + kernel_size * input_index_div2 + input_index_mod2 * ((1 + kernel_size) // 2)), reduction_mask & index_within_bounds & input_mask, eviction_policy='evict_first', other=0.0)
        squared_value = loaded_value * loaded_value
        zero_filled = tl.full(squared_value.shape, 0, squared_value.dtype)
        conditional_squared = tl.where(index_within_bounds, squared_value, zero_filled)
        broadcasted_squared = tl.broadcast_to(conditional_squared, [XBLOCK, RBLOCK])
        temp_sum_update = temp_sum + broadcasted_squared
        temp_sum = tl.where(reduction_mask & input_mask, temp_sum_update, temp_sum)

    summed_values = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (input_index), summed_values, input_mask)