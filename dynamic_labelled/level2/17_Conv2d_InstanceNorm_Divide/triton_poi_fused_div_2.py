# From: 17_Conv2d_InstanceNorm_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_2poi_fused_div_2(input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x2 = x_index
    x1 = x_index // kernel_size0
    input_value0 = tl.load(input_ptr0 + (x2), x_mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (x1), x_mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (x1), x_mask, eviction_policy='evict_last')
    subtracted_value = input_value0 - input_value1
    divisor_constant = 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1)
    divisor_float = divisor_constant.to(tl.float32)
    normalized_value = input_value2 / divisor_float
    epsilon = 1e-05
    stabilized_value = normalized_value + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(stabilized_value)
    scaled_value = subtracted_value * reciprocal_sqrt
    scale_factor = 0.5
    final_value = scaled_value * scale_factor
    tl.store(output_ptr0 + (x2), final_value, x_mask)