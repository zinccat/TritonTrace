# From: 37_FrobeniusNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_linalg_vector_norm_1per_fused_linalg_vector_norm_1(input_ptr, output_ptr, x_num_elements, r_num_elements):
    X_BLOCK: tl.constexpr = 1
    r_num_elements = 328
    R_BLOCK: tl.constexpr = 512
    x_offset = tl.program_id(0) * X_BLOCK
    tl.full([1], x_offset, tl.int32)
    tl.full([R_BLOCK], True, tl.int1)
    r_index = tl.arange(0, R_BLOCK)[:]
    r_mask = r_index < r_num_elements
    r_indices = r_index
    loaded_values = tl.load(input_ptr + (r_indices), r_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [R_BLOCK])
    masked_values = tl.where(r_mask, broadcasted_values, 0)
    sum_result = triton_helpers.promote_to_tensor(tl.sum(masked_values, 0))
    tl.store(output_ptr + (tl.full([1], 0, tl.int32)), sum_result, None)