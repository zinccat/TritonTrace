# From: 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_convolution_hardtanh_mish_mul_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    input_index = xindex
    weight_index = (xindex // 1089) % 64
    
    # Load data
    input_value = tl.load(in_out_ptr0 + (input_index), None)
    weight_value = tl.load(in_ptr0 + (weight_index), None, eviction_policy='evict_last')
    
    # Perform addition
    added_value = input_value + weight_value
    
    # Mish activation function
    threshold = 20.0
    is_greater_than_threshold = added_value > threshold
    exp_value = tl.math.exp(added_value)
    log1p_value = tl.extra.cuda.libdevice.log1p(exp_value)
    mish_value = tl.where(is_greater_than_threshold, added_value, log1p_value)
    
    # Hardtanh activation function
    tanh_value = tl.extra.cuda.libdevice.tanh(mish_value)
    mish_tanh_product = added_value * tanh_value
    half_value = 0.5
    mish_tanh_shifted = mish_tanh_product + half_value
    lower_bound = -1.0
    upper_bound = 1.0
    clamped_value = triton_helpers.maximum(mish_tanh_shifted, lower_bound)
    hardtanh_value = triton_helpers.minimum(clamped_value, upper_bound)
    scaled_hardtanh_value = hardtanh_value * 2.0
    
    # Store results
    tl.store(in_out_ptr0 + (input_index), added_value, None)
    tl.store(out_ptr0 + (input_index), scaled_hardtanh_value, None)