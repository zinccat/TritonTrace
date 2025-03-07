# From: 23_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__softmax_0(input_ptr, output_ptr, num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    num_elements = 32
    reduction_num_elements = 8192
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    max_values = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = reduction_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_1 = r_indices
        loaded_values = tl.load(input_ptr + (r_indices_1 + (8192 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        max_values_update = triton_helpers.maximum(max_values, broadcasted_values)
        max_values = tl.where(r_mask & x_mask, max_values_update, max_values)
    
    max_values_broadcast = triton_helpers.max2(max_values, 1)[:, None]
    tl.store(output_ptr + (x_indices_0), max_values_broadcast, x_mask)