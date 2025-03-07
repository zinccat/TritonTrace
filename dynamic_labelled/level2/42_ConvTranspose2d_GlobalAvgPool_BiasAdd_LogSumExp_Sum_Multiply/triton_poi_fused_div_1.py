# From: 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_1poi_fused_div_1(input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    div_kernel_size0 = index // kernel_size0
    div_kernel_size1 = index // kernel_size1
    original_index = index
    loaded_value0 = tl.load(input_ptr0 + (div_kernel_size0), mask, eviction_policy='evict_last')
    loaded_value1 = tl.load(input_ptr1 + (div_kernel_size1), mask, eviction_policy='evict_last')
    constant_multiplier = 10.0
    multiplied_value = loaded_value0 * constant_multiplier
    multiplied_result = multiplied_value * loaded_value1
    divisor = kernel_size1
    float_divisor = divisor.to(tl.float32)
    result = multiplied_result / float_divisor
    tl.store(output_ptr0 + (original_index), result, mask)