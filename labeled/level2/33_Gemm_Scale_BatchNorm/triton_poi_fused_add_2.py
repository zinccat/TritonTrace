# From: 33_Gemm_Scale_BatchNorm

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_2(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, XBLOCK)[:]
    mask = tl.full([XBLOCK], True, tl.int1)
    
    input_value = tl.load(in_ptr0 + (0))
    broadcasted_input = tl.broadcast_to(input_value, [XBLOCK])
    increment_value = tl.full([1], 1, tl.int64)
    
    result = broadcasted_input + increment_value
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), result, None)