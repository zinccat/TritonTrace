# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_layer_norm_2poi_fused_native_layer_norm_2(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    element_index = block_indices
    element_mod = element_index % 3136
    element_div_large = element_index // 301056
    element_div_small = (element_index // 3136) % 96
    
    input_value0 = tl.load(input_ptr0 + (element_index), None)
    input_value1 = tl.load(input_ptr1 + (element_mod + 3136 * element_div_large), None, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (element_mod + 3136 * element_div_large), None, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + (element_div_small), None, eviction_policy='evict_last')
    input_value4 = tl.load(input_ptr4 + (element_div_small), None, eviction_policy='evict_last')
    
    intermediate1 = input_value0 - input_value1
    intermediate2 = intermediate1 * input_value2
    intermediate3 = intermediate2 * input_value3
    result = intermediate3 + input_value4
    
    tl.store(output_ptr0 + (element_index), result, None)