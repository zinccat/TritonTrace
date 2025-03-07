# From: 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_0(input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    input_index1 = index // kernel_size0
    output_index = index
    loaded_value0 = tl.load(input_ptr0 + (input_index1), mask, eviction_policy='evict_last').to(tl.int1)
    loaded_value1 = tl.load(input_ptr1 + (input_index1), mask, eviction_policy='evict_last')
    default_value = 0.0
    selected_value = tl.where(loaded_value0, loaded_value1, default_value)
    divisor = (-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2)
    divisor_float = divisor.to(tl.float32)
    result = selected_value / divisor_float
    tl.store(output_ptr0 + (output_index), result, mask)