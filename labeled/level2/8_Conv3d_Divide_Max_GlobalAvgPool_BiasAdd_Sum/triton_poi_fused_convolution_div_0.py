# From: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_div_0(input_ptr0, input_ptr1, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    global_index = index
    batch_index = (index // 12600) % 16
    spatial_index = index % 12600
    batch_offset = (index // 12600)
    
    input_value0 = tl.load(input_ptr0 + (global_index), None)
    input_value1 = tl.load(input_ptr1 + (batch_index), None, eviction_policy='evict_last')
    
    sum_values = input_value0 + input_value1
    scale_factor = 0.5
    scaled_value = sum_values * scale_factor
    
    output_index = spatial_index + (12608 * batch_offset)
    tl.store(output_ptr0 + (output_index), scaled_value, None)