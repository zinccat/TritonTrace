# From: 39_L2Norm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_linalg_vector_norm_0red_fused_linalg_vector_norm_0(in_ptr0, out_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_0 = (input_indices % 2)
    input_index_1 = input_indices // 2
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_indices

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_index_2 = reduction_indices
        temp_index_0 = reduction_index_2 + input_index_0 * ((1 + kernel_size) // 2)
        temp_index_1 = kernel_size
        temp_mask_2 = temp_index_0 < temp_index_1
        loaded_values = tl.load(in_ptr0 + (reduction_index_2 + kernel_size * input_index_1 + input_index_0 * ((1 + kernel_size) // 2)), reduction_mask & temp_mask_2 & input_mask, eviction_policy='evict_first', other=0.0)
        squared_values = loaded_values * loaded_values
        zero_filled = tl.full(squared_values.shape, 0, squared_values.dtype)
        masked_squared_values = tl.where(temp_mask_2, squared_values, zero_filled)
        broadcasted_values = tl.broadcast_to(masked_squared_values, [XBLOCK, RBLOCK])
        temp_accumulator = temp_accumulator + broadcasted_values
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulator, temp_accumulator)

    summed_values = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (input_index_3), summed_values, input_mask)