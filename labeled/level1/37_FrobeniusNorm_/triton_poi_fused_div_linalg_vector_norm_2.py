# From: 37_FrobeniusNorm_

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_div_linalg_vector_norm_2(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    indices = block_indices
    input_values = tl.load(input_ptr0 + (indices), None)
    scalar_value = tl.load(input_ptr1 + (0))
    broadcasted_scalar = tl.broadcast_to(scalar_value, [BLOCK_SIZE])
    sqrt_scalar = tl.extra.cuda.libdevice.sqrt(broadcasted_scalar)
    result_values = input_values / sqrt_scalar
    tl.store(output_ptr0 + (indices), result_values, None)