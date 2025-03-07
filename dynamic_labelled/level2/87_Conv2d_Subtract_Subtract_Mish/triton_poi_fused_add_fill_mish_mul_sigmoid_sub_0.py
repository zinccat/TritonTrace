# From: 87_Conv2d_Subtract_Subtract_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_mish_mul_sigmoid_sub_0poi_fused_add_fill_mish_mul_sigmoid_sub_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    
    input_value = tl.load(in_ptr0 + (x0), xmask)
    subtracted_value = tl.load(in_out_ptr0 + (x0), xmask)
    
    half = 0.5
    subtracted_half = subtracted_value - half
    
    two_tenths = 0.2
    subtracted_two_tenths = subtracted_half - two_tenths
    
    threshold = 20.0
    is_greater_than_threshold = subtracted_two_tenths > threshold
    
    exp_value = tl.math.exp(subtracted_two_tenths)
    log1p_exp_value = tl.extra.cuda.libdevice.log1p(exp_value)
    
    mish_value = tl.where(is_greater_than_threshold, subtracted_two_tenths, log1p_exp_value)
    tanh_mish_value = tl.extra.cuda.libdevice.tanh(mish_value)
    
    sigmoid_value = tl.sigmoid(subtracted_two_tenths)
    product_subtracted_sigmoid = subtracted_two_tenths * sigmoid_value
    
    squared_tanh_mish = tanh_mish_value * tanh_mish_value
    one = 1.0
    one_minus_squared_tanh_mish = one - squared_tanh_mish
    
    mish_result = product_subtracted_sigmoid * one_minus_squared_tanh_mish
    final_mish_value = tanh_mish_value + mish_result
    
    output_value = input_value * final_mish_value
    
    tl.store(in_out_ptr0 + (x0), output_value, xmask)