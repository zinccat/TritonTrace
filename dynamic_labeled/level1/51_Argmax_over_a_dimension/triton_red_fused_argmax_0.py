# From: 51_Argmax_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_argmax_0(in_ptr0, out_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_mod_kernel = input_index % kernel_size
    input_div_kernel = input_index // kernel_size
    max_values = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    max_indices = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    original_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_flat = reduction_index
        loaded_values = tl.load(in_ptr0 + (input_mod_kernel + kernel_size * reduction_index_flat + input_div_kernel * kernel_size * kernel_size), reduction_mask & input_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        max_values_next, max_indices_next = triton_helpers.maximum_with_index(
            max_values, max_indices, broadcasted_values, reduction_index
        )
        max_values = tl.where(reduction_mask & input_mask, max_values_next, max_values)
        max_indices = tl.where(reduction_mask & input_mask, max_indices_next, max_indices)

    final_max_indices, _ = triton_helpers.max_with_index(max_values, max_indices, 1)
    final_indices = final_max_indices[:, None]
    tl.store(out_ptr0 + (original_index), final_indices, input_mask)