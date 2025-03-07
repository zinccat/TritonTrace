# From: 44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_mul_0(input_ptr, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    input_index = indices // kernel_size
    output_index = indices
    loaded_values = tl.load(input_ptr + (input_index), mask, eviction_policy='evict_last')
    multiplier = 1.0
    multiplied_values = loaded_values * multiplier
    kernel_size_float = kernel_size.to(tl.float32)
    divided_values = multiplied_values / kernel_size_float
    scale_factor = 0.5
    scaled_values = divided_values * scale_factor
    tl.store(output_ptr + (output_index), scaled_values, mask)