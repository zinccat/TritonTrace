# From: 81_Gemm_Swish_Divide_Clamp_Tanh_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clamp_div_mul_sigmoid_tanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_values = tl.load(in_ptr0 + (x0), xmask)
    sigmoid_values = tl.sigmoid(input_values)
    elementwise_mul = input_values * sigmoid_values
    half = 0.5
    scaled_values = elementwise_mul * half
    neg_one = -1.0
    clamped_min = triton_helpers.maximum(scaled_values, neg_one)
    one = 1.0
    clamped_max = triton_helpers.minimum(clamped_min, one)
    tanh_values = tl.extra.cuda.libdevice.tanh(clamped_max)
    tanh_clamped_min = triton_helpers.maximum(tanh_values, neg_one)
    tanh_clamped_max = triton_helpers.minimum(tanh_clamped_min, one)
    tl.store(out_ptr0 + (x0), tanh_clamped_max, xmask)