# From: 22_Tanh

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_tanh_0(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    indices = block_indices
    input_values = tl.load(input_ptr + (indices), None)
    tanh_values = tl.extra.cuda.libdevice.tanh(input_values)
    tl.store(output_ptr + (indices), tanh_values, None)