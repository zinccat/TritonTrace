# From: 48_Mean_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_0(in_ptr0, out_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_mod0 = input_index % kernel_size0
    input_index_div0 = input_index // kernel_size0
    input_index_mod1 = input_index // kernel_size1
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    original_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_value = reduction_index
        temp_index = reduction_index_value + input_index_div0 * ((1 + kernel_size0) // 2)
        kernel_size0_mask = temp_index < kernel_size0
        temp_load = tl.load(
            in_ptr0 + (input_index_mod0 + kernel_size0 * reduction_index_value + input_index_mod1 * kernel_size0 * kernel_size0 + kernel_size0 * input_index_div0 * ((1 + kernel_size0) // 2)),
            reduction_mask & kernel_size0_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_accumulate = temp_sum + temp_broadcast
        temp_sum = tl.where(reduction_mask & input_mask, temp_accumulate, temp_sum)

    temp_result = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (original_index), temp_result, input_mask)