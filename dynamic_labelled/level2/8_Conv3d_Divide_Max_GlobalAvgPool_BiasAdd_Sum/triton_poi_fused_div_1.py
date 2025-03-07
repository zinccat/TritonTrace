# From: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_1poi_fused_div_1(input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    z_index = index // kernel_size_x
    x_index = index
    input_value = tl.load(input_ptr + (z_index), mask, eviction_policy='evict_last')
    
    divisor = (-1) + ((-1) * (kernel_size_x // 2) * (kernel_size_x // 2)) + 2 * (kernel_size_x // 2) + (kernel_size_x // 2) * (kernel_size_x // 2) * (kernel_size_y // 2) + ((-2) * (kernel_size_y // 2) * (kernel_size_x // 2)) + (kernel_size_y // 2)
    divisor_float = divisor.to(tl.float32)
    
    result = input_value / divisor_float
    tl.store(output_ptr + (x_index), result, mask)