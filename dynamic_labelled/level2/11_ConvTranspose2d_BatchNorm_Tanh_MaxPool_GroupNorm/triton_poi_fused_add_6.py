# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_6(input_ptr, output_ptr, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, XBLOCK)
    mask = tl.full([XBLOCK], True, tl.int1)
    
    input_value = tl.load(input_ptr + (0))
    broadcasted_input = tl.broadcast_to(input_value, [XBLOCK])
    increment_value = tl.full([1], 1, tl.int64)
    
    result = broadcasted_input + increment_value
    tl.store(output_ptr + (tl.full([XBLOCK], 0, tl.int32)), result, None)