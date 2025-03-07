# From: 17_Conv2d_InstanceNorm_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_2(input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x2 = x_index
    x1 = x_index // kernel_size0
    input_value0 = tl.load(input_ptr0 + (x2), x_mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (x1), x_mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (x1), x_mask, eviction_policy='evict_last')
    subtracted_value = input_value0 - input_value1
    constant_term = 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1)
    constant_term_float = constant_term.to(tl.float32)
    division_result = input_value2 / constant_term_float
    epsilon = 1e-05
    adjusted_division = division_result + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_division)
    multiplied_value = subtracted_value * reciprocal_sqrt
    scale_factor = 0.5
    scaled_result = multiplied_value * scale_factor
    tl.store(output_ptr0 + (x2), scaled_result, x_mask)