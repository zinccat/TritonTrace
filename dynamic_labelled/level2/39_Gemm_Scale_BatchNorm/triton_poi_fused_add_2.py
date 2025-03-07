# From: 39_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_2poi_fused_add_2(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, XBLOCK)[:]
    mask = tl.full([XBLOCK], True, tl.int1)
    
    input_data = tl.load(in_ptr0 + (0))
    broadcasted_data = tl.broadcast_to(input_data, [XBLOCK])
    increment_value = tl.full([1], 1, tl.int64)
    
    result_data = broadcasted_data + increment_value
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), result_data, None)