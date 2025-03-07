# From: 25_Conv2d_Min_Tanh_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_tanh_tanh_backward_1poi_fused_tanh_tanh_backward_1(input_ptr, output_ptr1, output_ptr2, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    indices = block_indices

    input_values = tl.load(input_ptr + (indices), mask)
    tanh_output1 = tl.extra.cuda.libdevice.tanh(input_values)
    tanh_output2 = tl.extra.cuda.libdevice.tanh(tanh_output1)
    tanh_squared = tanh_output1 * tanh_output1
    one = 1.0
    derivative = one - tanh_squared

    tl.store(output_ptr1 + (indices), tanh_output2, mask)
    tl.store(output_ptr2 + (indices), derivative, mask)