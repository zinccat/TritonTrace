# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_1(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    input_data0 = tl.load(input_ptr0 + (indices), valid_mask)
    input_data1 = tl.load(input_ptr1 + (0))
    broadcasted_data1 = tl.broadcast_to(input_data1, [BLOCK_SIZE])
    multiplied_data = input_data0 * broadcasted_data1
    tl.store(output_ptr0 + (indices), multiplied_data, valid_mask)