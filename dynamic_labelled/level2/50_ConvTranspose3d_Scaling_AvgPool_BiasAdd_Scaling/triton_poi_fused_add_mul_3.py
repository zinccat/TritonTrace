# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_mul_3(input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    input_index = index
    pooled_index = ((index // kernel_size0) % 16)
    
    input_data0 = tl.load(input_ptr0 + (input_index), mask, eviction_policy='evict_last')
    input_data1 = tl.load(input_ptr1 + (pooled_index), mask, eviction_policy='evict_last')
    scaling_factor = tl.load(input_ptr2 + (0))
    broadcast_scaling = tl.broadcast_to(scaling_factor, [XBLOCK])
    
    added_data = input_data0 + input_data1
    scaled_data = added_data * broadcast_scaling
    
    tl.store(output_ptr0 + (input_index), scaled_data, mask)