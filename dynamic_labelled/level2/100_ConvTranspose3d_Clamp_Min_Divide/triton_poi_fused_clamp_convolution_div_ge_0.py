# From: 100_ConvTranspose3d_Clamp_Min_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clamp_convolution_div_ge_0poi_fused_clamp_convolution_div_ge_0(input_ptr0, input_ptr1, output_ptr0, output_ptr1, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    channel_index = (indices // kernel_size) % 16
    input_value0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (channel_index), mask, eviction_policy='evict_last')
    sum_values = input_value0 + input_value1
    clamp_min_value = -1.0
    clamped_value = triton_helpers.maximum(sum_values, clamp_min_value)
    scale_factor = 0.5
    scaled_value = clamped_value * scale_factor
    is_greater_equal = sum_values >= clamp_min_value
    tl.store(output_ptr0 + (linear_index), scaled_value, mask)
    tl.store(output_ptr1 + (linear_index), is_greater_equal, mask)