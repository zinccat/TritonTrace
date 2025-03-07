# From: 22_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_tanh_0poi_fused_tanh_0(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    input_values = tl.load(input_ptr + (indices), valid_mask)
    tanh_values = tl.extra.cuda.libdevice.tanh(input_values)
    tl.store(output_ptr + (indices), tanh_values, valid_mask)