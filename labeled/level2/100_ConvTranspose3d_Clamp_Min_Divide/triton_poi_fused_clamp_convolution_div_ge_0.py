# From: 100_ConvTranspose3d_Clamp_Min_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_clamp_convolution_div_ge_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 31497984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    linear_index = xindex
    block_index = (xindex // 123039) % 16
    sub_block_index = xindex % 3969
    major_index = (xindex // 3969)
    
    input_value0 = tl.load(in_ptr0 + (linear_index), xmask)
    input_value1 = tl.load(in_ptr1 + (block_index), xmask, eviction_policy='evict_last')
    
    sum_values = input_value0 + input_value1
    clamp_min_value = -1.0
    clamped_value = triton_helpers.maximum(sum_values, clamp_min_value)
    
    scale_factor = 0.5
    scaled_value = clamped_value * scale_factor
    
    is_greater_equal = sum_values >= clamp_min_value
    
    tl.store(out_ptr0 + (linear_index), scaled_value, xmask)
    tl.store(out_ptr1 + (sub_block_index + (4096 * major_index)), is_greater_equal, xmask)