# From: 8_ResNetBasicBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_4poi_fused_add_4(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, XBLOCK)[:]
    mask = tl.full([XBLOCK], True, tl.int1)
    input_value = tl.load(in_ptr0 + (0))
    broadcasted_input = tl.broadcast_to(input_value, [XBLOCK])
    increment_value = tl.full([1], 1, tl.int64)
    result = broadcasted_input + increment_value
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), result, None)