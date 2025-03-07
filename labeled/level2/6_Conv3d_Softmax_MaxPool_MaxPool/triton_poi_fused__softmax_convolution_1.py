# From: 6_Conv3d_Softmax_MaxPool_MaxPool

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__softmax_convolution_1(input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    block_index = index
    channel_index = (index // 12600) % 16
    spatial_index = index % 12600
    batch_index = (index // 201600)
    channel_block_index = (index // 12600)
    
    input_value0 = tl.load(input_ptr0 + (block_index), None)
    input_value1 = tl.load(input_ptr1 + (channel_index), None, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (spatial_index + (12608 * batch_index)), None, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + (spatial_index + (12608 * batch_index)), None, eviction_policy='evict_last')
    
    sum_input = input_value0 + input_value1
    difference = sum_input - input_value2
    exp_value = tl.math.exp(difference)
    softmax_output = exp_value / input_value3
    
    tl.store(output_ptr0 + (spatial_index + (12608 * channel_block_index)), softmax_output, None)