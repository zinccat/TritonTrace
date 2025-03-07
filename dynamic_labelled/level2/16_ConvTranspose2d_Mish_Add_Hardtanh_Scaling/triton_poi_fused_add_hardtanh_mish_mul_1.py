# From: 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_hardtanh_mish_mul_1poi_fused_add_hardtanh_mish_mul_1(input_ptr, output_ptr, scale_factor, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    input_indices = indices
    input_values = tl.load(input_ptr + (input_indices), mask)
    
    threshold = 20.0
    is_greater_than_threshold = input_values > threshold
    exp_values = tl.math.exp(input_values)
    log1p_values = tl.extra.cuda.libdevice.log1p(exp_values)
    mish_values = tl.where(is_greater_than_threshold, input_values, log1p_values)
    tanh_values = tl.extra.cuda.libdevice.tanh(mish_values)
    mish_tanh_product = input_values * tanh_values
    
    half = 0.5
    mish_tanh_sum = mish_tanh_product + half
    
    lower_bound = -1.0
    upper_bound = 1.0
    clamped_values = triton_helpers.maximum(mish_tanh_sum, lower_bound)
    clamped_values = triton_helpers.minimum(clamped_values, upper_bound)
    
    scale_factor_float = scale_factor.to(tl.float32)
    scaled_output = clamped_values * scale_factor_float
    
    tl.store(output_ptr + (input_indices), scaled_output, mask)