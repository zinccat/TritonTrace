# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_native_group_norm_3(input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    group_index = (index // kernel_size) % 16
    
    input_value0 = tl.load(input_ptr0 + (element_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (group_index), mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (group_index), mask, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + (group_index), mask, eviction_policy='evict_last')
    
    intermediate0 = input_value0 * input_value1
    intermediate1 = intermediate0 + input_value2
    result = intermediate1 * input_value3
    
    tl.store(output_ptr0 + (element_index), result, mask)