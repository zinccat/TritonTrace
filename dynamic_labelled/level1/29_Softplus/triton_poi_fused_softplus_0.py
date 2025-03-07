# From: 29_Softplus

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_softplus_0poi_fused_softplus_0(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    indices = block_indices
    input_values = tl.load(input_ptr + (indices), mask)
    threshold = 20.0
    is_greater_than_threshold = input_values > threshold
    exp_values = tl.math.exp(input_values)
    log1p_values = tl.extra.cuda.libdevice.log1p(exp_values)
    softplus_values = tl.where(is_greater_than_threshold, input_values, log1p_values + threshold)
    tl.store(output_ptr + (indices), softplus_values, mask)