# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_scatter_zeros_1(input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    mod_index = index % kernel_size0
    div_index = index // kernel_size0
    input_value0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (linear_index), mask, eviction_policy='evict_last')
    tl.device_assert(((0 <= input_value0) & (input_value0 < 16)) | ~mask, "index out of bounds: 0 <= input_value0 < 16")
    computed_index = (
        mod_index + 
        (-128) * div_index + 
        (-8) * input_value0 + 
        (-32) * div_index * kernel_size2 * kernel_size2 + 
        (-2) * input_value0 * kernel_size2 * kernel_size2 + 
        4 * kernel_size1 * input_value0 + 
        8 * kernel_size2 * input_value0 + 
        64 * kernel_size1 * div_index + 
        128 * kernel_size2 * div_index + 
        kernel_size1 * input_value0 * kernel_size2 * kernel_size2 + 
        (-64) * kernel_size1 * kernel_size2 * div_index + 
        (-4) * kernel_size1 * kernel_size2 * input_value0 + 
        16 * kernel_size1 * div_index * kernel_size2 * kernel_size2
    )
    tl.store(output_ptr0 + computed_index, input_value1, mask)