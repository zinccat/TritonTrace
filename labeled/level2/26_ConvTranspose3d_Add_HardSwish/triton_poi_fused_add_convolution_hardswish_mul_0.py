# From: 26_ConvTranspose3d_Add_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_convolution_hardswish_mul_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x3 = x_index
    x1 = (x_index // 32768) % 64
    
    input_out_value = tl.load(in_out_ptr0 + (x3), None)
    input_value0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    input_value1 = tl.load(in_ptr1 + (x3), None)
    
    sum_value = input_out_value + input_value0
    combined_sum = sum_value + input_value1
    
    bias_value = 3.0
    biased_sum = combined_sum + bias_value
    
    lower_bound = 0.0
    upper_bound = 6.0
    
    clamped_value = triton_helpers.minimum(triton_helpers.maximum(biased_sum, lower_bound), upper_bound)
    hardswish_value = combined_sum * clamped_value
    
    scale_factor = 0.16666666666666666
    scaled_hardswish = hardswish_value * scale_factor
    final_output = combined_sum * scaled_hardswish
    
    tl.store(in_out_ptr0 + (x3), combined_sum, None)
    tl.store(out_ptr0 + (x3), final_output, None)