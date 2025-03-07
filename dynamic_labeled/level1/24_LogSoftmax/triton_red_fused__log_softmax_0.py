# From: 24_LogSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_0(in_ptr0, out_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_0 = (input_index % 2)
    input_index_1 = input_index // 2
    max_values = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index
        temp_index_0 = reduction_index_2 + input_index_0 * ((1 + kernel_size) // 2)
        temp_index_1 = kernel_size
        temp_mask_2 = temp_index_0 < temp_index_1
        temp_values_3 = tl.load(
            in_ptr0 + (reduction_index_2 + kernel_size * input_index_1 + input_index_0 * ((1 + kernel_size) // 2)),
            reduction_mask & temp_mask_2 & input_mask,
            eviction_policy='evict_first',
            other=float("-inf")
        )
        temp_broadcast_4 = tl.broadcast_to(temp_values_3, [XBLOCK, RBLOCK])
        temp_max_6 = triton_helpers.maximum(max_values, temp_broadcast_4)
        max_values = tl.where(reduction_mask & input_mask, temp_max_6, max_values)

    max_values_5 = triton_helpers.max2(max_values, 1)[:, None]
    tl.store(out_ptr0 + (input_index_3), max_values_5, input_mask)