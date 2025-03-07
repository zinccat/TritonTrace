# From: 32_Conv2d_Scaling_Min

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_scatter_zeros_1poi_fused_scatter_zeros_1(input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    column_index = index % kernel_size0
    row_index = index // kernel_size0
    
    input_value0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (linear_index), mask, eviction_policy='evict_last')
    
    tl.device_assert(((0 <= input_value0) & (input_value0 < 16)) | ~mask, "index out of bounds: 0 <= input_value0 < 16")
    
    output_index = (
        column_index + 4 * input_value0 + 64 * row_index +
        input_value0 * kernel_size1 * kernel_size1 +
        (-64) * kernel_size1 * row_index +
        (-4) * kernel_size1 * input_value0 +
        16 * row_index * kernel_size1 * kernel_size1
    )
    
    tl.store(output_ptr0 + output_index, input_value1, mask)