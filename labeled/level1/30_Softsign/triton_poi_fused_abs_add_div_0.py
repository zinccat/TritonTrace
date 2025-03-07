# From: 30_Softsign

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_abs_add_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    input_index = xindex
    input_value = tl.load(in_ptr0 + (input_index), None)
    absolute_value = tl.math.abs(input_value)
    constant_one = 1.0
    sum_with_one = absolute_value + constant_one
    result = input_value / sum_with_one
    tl.store(out_ptr0 + (input_index), result, None)