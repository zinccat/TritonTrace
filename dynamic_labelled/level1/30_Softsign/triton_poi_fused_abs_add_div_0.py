# From: 30_Softsign

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_add_div_0poi_fused_abs_add_div_0(in_ptr0, out_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    input_indices = indices
    input_values = tl.load(in_ptr0 + (input_indices), mask)
    abs_values = tl.math.abs(input_values)
    one = 1.0
    sum_values = abs_values + one
    result_values = input_values / sum_values
    tl.store(out_ptr0 + (input_indices), result_values, mask)