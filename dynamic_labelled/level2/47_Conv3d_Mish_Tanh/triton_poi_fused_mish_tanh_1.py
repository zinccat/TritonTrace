# From: 47_Conv3d_Mish_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mish_tanh_1poi_fused_mish_tanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_value = tl.load(in_ptr0 + (x0), xmask)
    threshold = 20.0
    is_greater_than_threshold = input_value > threshold
    exp_input = tl.math.exp(input_value)
    log1p_exp_input = tl.extra.cuda.libdevice.log1p(exp_input)
    mish_result = tl.where(is_greater_than_threshold, input_value, log1p_exp_input)
    tanh_mish = tl.extra.cuda.libdevice.tanh(mish_result)
    product = input_value * tanh_mish
    tanh_product = tl.extra.cuda.libdevice.tanh(product)
    tl.store(out_ptr0 + (x0), tanh_product, xmask)