# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_5(input_ptr0, input_ptr1, output_ptr0, kernel_size0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    group_index = (index // kernel_size0) % 16
    loaded_input0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    loaded_input1 = tl.load(input_ptr1 + (group_index), mask, eviction_policy='evict_last')
    multiplied_result = loaded_input0 * loaded_input1
    tl.store(output_ptr0 + (linear_index), multiplied_result, mask)