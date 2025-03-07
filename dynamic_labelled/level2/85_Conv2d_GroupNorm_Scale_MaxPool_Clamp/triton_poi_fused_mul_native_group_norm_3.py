# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_native_group_norm_3poi_fused_mul_native_group_norm_3(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size) % 16)
    
    input_val0 = tl.load(input_ptr0 + (x3), x_mask, eviction_policy='evict_last')
    input_val1 = tl.load(input_ptr1 + (x1), x_mask, eviction_policy='evict_last')
    input_val2 = tl.load(input_ptr2 + (x1), x_mask, eviction_policy='evict_last')
    input_val3 = tl.load(input_ptr3 + (x1), x_mask, eviction_policy='evict_last')
    
    intermediate_val1 = input_val0 * input_val1
    intermediate_val2 = intermediate_val1 + input_val2
    output_val = intermediate_val2 * input_val3
    
    tl.store(output_ptr0 + (x3), output_val, x_mask)