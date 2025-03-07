# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_25poi_fused_convolution_25(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = x_index
    x1 = ((x_index // 32768) % 4)
    output_value = tl.load(in_out_ptr0 + (x3), None)
    input_value = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    result_value = output_value + input_value
    tl.store(in_out_ptr0 + (x3), result_value, None)