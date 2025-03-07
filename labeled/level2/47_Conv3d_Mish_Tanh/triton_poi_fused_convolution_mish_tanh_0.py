# From: 47_Conv3d_Mish_Tanh

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_mish_tanh_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = x_index
    x1 = (x_index // 12600) % 16
    input_value = tl.load(in_out_ptr0 + (x3), None)
    weight_value = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    sum_value = input_value + weight_value
    threshold = 20.0
    is_greater_than_threshold = sum_value > threshold
    exp_value = tl.math.exp(sum_value)
    log1p_value = tl.extra.cuda.libdevice.log1p(exp_value)
    mish_value = tl.where(is_greater_than_threshold, sum_value, log1p_value)
    tanh_mish = tl.extra.cuda.libdevice.tanh(mish_value)
    product_value = sum_value * tanh_mish
    tanh_product = tl.extra.cuda.libdevice.tanh(product_value)
    tl.store(in_out_ptr0 + (x3), sum_value, None)
    tl.store(out_ptr0 + (x3), tanh_product, None)