# From: 52_Argmin_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_argmin_0(in_ptr0, out_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_dim0 = (input_index % kernel_size)
    input_dim1 = input_index // kernel_size
    min_values = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    min_indices = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    original_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_dim = reduction_index
        loaded_values = tl.load(in_ptr0 + (input_dim0 + kernel_size * reduction_dim + input_dim1 * kernel_size * kernel_size), reduction_mask & input_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        min_values_next, min_indices_next = triton_helpers.minimum_with_index(
            min_values, min_indices, broadcasted_values, reduction_index
        )
        min_values = tl.where(reduction_mask & input_mask, min_values_next, min_values)
        min_indices = tl.where(reduction_mask & input_mask, min_indices_next, min_indices)

    min_value, min_index = triton_helpers.min_with_index(min_values, min_indices, 1)
    min_index = min_index[:, None]
    tl.store(out_ptr0 + (original_index), min_index, input_mask)