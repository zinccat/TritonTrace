# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_scatter_zeros_1poi_fused_scatter_zeros_1(input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    kernel_index0 = index % kernel_size0
    kernel_index1 = index // kernel_size0
    temp0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    temp2 = tl.load(input_ptr1 + (linear_index), mask, eviction_policy='evict_last')
    tl.device_assert(((0 <= temp0) & (temp0 < 16)) | ~mask, "index out of bounds: 0 <= temp0 < 16")
    tl.store(output_ptr0 + (kernel_index0 + kernel_size1 * temp0 * kernel_size2 * kernel_size2 + 16 * kernel_size1 * kernel_index1 * kernel_size2 * kernel_size2), temp2, mask)