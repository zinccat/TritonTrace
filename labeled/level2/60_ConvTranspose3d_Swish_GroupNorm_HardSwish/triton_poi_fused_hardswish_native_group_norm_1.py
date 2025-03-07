# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_hardswish_native_group_norm_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr1, xnumel, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x_index_5 = x_index
    x_index_7 = (x_index // 123039)
    x_index_2 = (x_index // 123039) % 16
    
    # Unused calculations
    x_index_mod_3969 = xindex % 3969
    x_index_div_3969 = (xindex // 3969)
    
    input_val0 = tl.load(input_ptr0 + (x_index_5), None)
    input_val1 = tl.load(input_ptr1 + ((x_index_7 // 4)), None, eviction_policy='evict_last')
    input_val2 = tl.load(input_ptr2 + ((x_index_7 // 4)), None, eviction_policy='evict_last')
    input_val3 = tl.load(input_ptr3 + (x_index_2), None, eviction_policy='evict_last')
    input_val4 = tl.load(input_ptr4 + (x_index_2), None, eviction_policy='evict_last')
    
    sigmoid_val = tl.sigmoid(input_val0)
    product_val = sigmoid_val * input_val0
    diff_val = product_val - input_val1
    product_val2 = diff_val * input_val2
    product_val3 = product_val2 * input_val3
    sum_val = product_val3 + input_val4
    
    constant_val1 = 3.0
    sum_with_constant = sum_val + constant_val1
    constant_val2 = 0.0
    max_val = triton_helpers.maximum(sum_with_constant, constant_val2)
    constant_val3 = 6.0
    min_val = triton_helpers.minimum(max_val, constant_val3)
    
    product_val4 = sum_val * min_val
    constant_val4 = 0.16666666666666666
    final_val = product_val4 * constant_val4
    
    tl.store(output_ptr1 + (x_index_5), final_val, None)