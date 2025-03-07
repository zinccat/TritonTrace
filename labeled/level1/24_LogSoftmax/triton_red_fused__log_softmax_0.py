# From: 24_LogSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__log_softmax_0(input_ptr, output_ptr, num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    num_elements = 32
    reduction_num_elements = 8192
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    _max_values = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = reduction_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_flat = r_indices
        loaded_values = tl.load(input_ptr + (r_indices_flat + (8192 * x_indices_flat)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        max_values = triton_helpers.maximum(_max_values, broadcasted_values)
        _max_values = tl.where(r_mask & x_mask, max_values, _max_values)
    
    max_values_across_reduction = triton_helpers.max2(_max_values, 1)[:, None]
    tl.store(output_ptr + (x_indices_flat), max_values_across_reduction, x_mask)