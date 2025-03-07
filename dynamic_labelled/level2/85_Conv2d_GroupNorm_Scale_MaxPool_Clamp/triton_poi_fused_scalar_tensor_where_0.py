# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_scalar_tensor_where_0(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    input_values0 = tl.load(input_ptr0 + (indices), valid_mask).to(tl.int1)
    input_values1 = tl.load(input_ptr1 + (indices), valid_mask)
    default_value = 0.0
    result_values = tl.where(input_values0, input_values1, default_value)
    tl.store(output_ptr0 + (indices), result_values, valid_mask)