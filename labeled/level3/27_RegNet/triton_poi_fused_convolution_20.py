# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_20poi_fused_convolution_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = x_index
    x0 = (x_index % 256)
    temp0 = tl.load(in_out_ptr0 + (x2), None)
    temp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    temp2 = temp0 + temp1
    tl.store(in_out_ptr0 + (x2), temp2, None)