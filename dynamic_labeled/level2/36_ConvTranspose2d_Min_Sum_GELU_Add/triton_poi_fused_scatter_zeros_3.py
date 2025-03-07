# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_scatter_zeros_3(input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    kernel_index0 = index % kernel_size0
    kernel_index2 = index // kernel_size1
    kernel_index1 = index % kernel_size1
    
    value0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    value2 = tl.load(input_ptr1 + (kernel_index0 + 2 * kernel_size2 * kernel_index2), mask, eviction_policy='evict_last')
    value3 = tl.load(input_ptr2 + (kernel_index0 + 2 * kernel_size2 * kernel_index2), mask, eviction_policy='evict_last')
    
    tl.device_assert(((0 <= value0) & (value0 < 16)) | ~mask, "index out of bounds: 0 <= value0 < 16")
    
    result = value2 * value3
    tl.store(output_ptr0 + (kernel_index1 + 4 * value0 * kernel_size2 * kernel_size2 + 64 * kernel_index2 * kernel_size2 * kernel_size2), result, mask)