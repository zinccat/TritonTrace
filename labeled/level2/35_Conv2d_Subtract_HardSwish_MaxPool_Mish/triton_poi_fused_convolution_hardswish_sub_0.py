# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_convolution_hardswish_sub_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    batch_index = (xindex // 900) % 16
    
    # Load data
    in_out_data = tl.load(in_out_ptr0 + (xindex), None)
    in_data = tl.load(in_ptr0 + (batch_index), None, eviction_policy='evict_last')
    
    # Perform addition
    added_data = in_out_data + in_data
    
    # HardSwish operation
    half = 0.5
    shifted_data = added_data - half
    three = 3.0
    shifted_plus_three = shifted_data + three
    zero = 0.0
    max_value = triton_helpers.maximum(shifted_plus_three, zero)
    six = 6.0
    min_value = triton_helpers.minimum(max_value, six)
    hardswish_result = shifted_data * min_value
    
    # Scale result
    scale_factor = 0.16666666666666666
    scaled_result = hardswish_result * scale_factor
    
    # Store results
    tl.store(in_out_ptr0 + (xindex), added_data, None)
    tl.store(out_ptr0 + (xindex), scaled_result, None)