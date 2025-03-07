# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_4poi_fused_add_4(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = tl.arange(0, BLOCK_SIZE)
    valid_mask = tl.full([BLOCK_SIZE], True, tl.int1)
    
    input_data = tl.load(input_ptr + (0))
    broadcasted_data = tl.broadcast_to(input_data, [BLOCK_SIZE])
    increment_value = tl.full([1], 1, tl.int64)
    result_data = broadcasted_data + increment_value
    
    tl.store(output_ptr + (tl.full([BLOCK_SIZE], 0, tl.int32)), result_data, None)