# From: 29_Softplus

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_softplus_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    input_indices = block_indices
    input_values = tl.load(in_ptr0 + (input_indices), None)
    threshold = 20.0
    is_greater_than_threshold = input_values > threshold
    exp_values = tl.math.exp(input_values)
    log1p_values = tl.extra.cuda.libdevice.log1p(exp_values)
    softplus_values = tl.where(is_greater_than_threshold, input_values, log1p_values + threshold)
    tl.store(out_ptr0 + (input_indices), softplus_values, None)