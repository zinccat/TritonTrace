# From: 48_Mean_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_0red_fused_mean_0(in_ptr0, out_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = ((input_index // kernel_size0) % 2)
    input_index_0 = (input_index % kernel_size0)
    input_index_2 = input_index // kernel_size1
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_4 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_3 = reduction_index
        temp_index_0 = reduction_index_3 + input_index_1 * ((1 + kernel_size0) // 2)
        temp_index_1 = kernel_size0
        temp_index_2 = temp_index_0 < temp_index_1
        loaded_value = tl.load(
            in_ptr0 + (input_index_0 + kernel_size0 * reduction_index_3 + input_index_2 * kernel_size0 * kernel_size0 + kernel_size0 * input_index_1 * ((1 + kernel_size0) // 2)),
            reduction_mask & temp_index_2 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        broadcasted_value = tl.broadcast_to(loaded_value, [XBLOCK, RBLOCK])
        temp_sum_updated = temp_sum + broadcasted_value
        temp_sum = tl.where(reduction_mask & input_mask, temp_sum_updated, temp_sum)

    summed_values = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (input_index_4), summed_values, input_mask)