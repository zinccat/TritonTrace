# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_4poi_fused_add_4(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = tl.arange(0, BLOCK_SIZE)
    mask = tl.full([BLOCK_SIZE], True, tl.int1)
    input_value = tl.load(input_ptr + (0))
    broadcasted_input = tl.broadcast_to(input_value, [BLOCK_SIZE])
    increment = tl.full([1], 1, tl.int64)
    result = broadcasted_input + increment
    tl.store(output_ptr + (tl.full([BLOCK_SIZE], 0, tl.int32)), result, None)