# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_3(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    
    input_values = tl.load(in_out_ptr0 + (x0), xmask)
    add_constant = 3.0
    added_values = input_values + add_constant
    zero_constant = 0.0
    max_values = triton_helpers.maximum(added_values, zero_constant)
    upper_bound = 6.0
    clamped_values = triton_helpers.minimum(max_values, upper_bound)
    multiplied_values = input_values * clamped_values
    scale_factor = 0.16666666666666666
    scaled_values = multiplied_values * scale_factor
    
    tl.store(in_out_ptr0 + (x0), scaled_values, xmask)