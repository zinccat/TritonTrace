# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_4(input_ptr, output_ptr, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, XBLOCK)[:]
    mask = tl.full([XBLOCK], True, tl.int1)
    input_value = tl.load(input_ptr + (0))
    broadcasted_input = tl.broadcast_to(input_value, [XBLOCK])
    increment_value = tl.full([1], 1, tl.int64)
    result = broadcasted_input + increment_value
    tl.store(output_ptr + (tl.full([XBLOCK], 0, tl.int32)), result, None)