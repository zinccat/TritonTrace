# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_0(input_ptr, output_ptr, seed_offset, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    indices = block_indices
    seed_value = tl.load(input_ptr + seed_offset)
    indices_as_uint32 = indices
    random_values = tl.rand(seed_value, indices_as_uint32.to(tl.uint32))
    dropout_threshold = 0.2
    dropout_mask = random_values > dropout_threshold
    tl.store(output_ptr + indices, dropout_mask, mask)