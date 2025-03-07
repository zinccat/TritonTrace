# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_7poi_fused_mul_7(in_out_ptr0, in_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    output_values = tl.load(in_out_ptr0 + (indices), valid_mask)
    input_value = tl.load(in_ptr0 + (0))
    broadcasted_value = tl.broadcast_to(input_value, [BLOCK_SIZE])
    result_values = output_values * broadcasted_value
    tl.store(in_out_ptr0 + (indices), result_values, valid_mask)